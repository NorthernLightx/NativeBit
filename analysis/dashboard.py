"""NativeBit TPU Training Dashboard — real-time monitoring with auto-sync.

Serves an interactive Plotly.js dashboard on localhost. In TPU mode,
a background thread syncs logs from the TPU VM every 30s via SCP.

Usage:
    # TPU sync mode (pulls logs from TPU VM)
    python analysis/dashboard.py \
        --tpu-name nativebit-v6e \
        --tpu-zone europe-west4-a \
        --tpu-project REDACTED_PROJECT_ID

    # Local mode (reads existing log files)
    python analysis/dashboard.py --log-dir logs/tpu
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> tuple[dict | None, list[dict]]:
    """Load a JSONL log file. Returns (header, records)."""
    header = None
    records = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if obj.get("type") == "header":
                    header = obj
                else:
                    records.append(obj)
    except (json.JSONDecodeError, OSError):
        pass
    return header, records


def load_results(path: Path) -> list[dict]:
    """Load overnight_results.json."""
    if not path.exists():
        return []
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def n_codebook_to_bits(n: int) -> int:
    """Map codebook entries to bit width."""
    return {4: 2, 8: 3, 16: 4, 32: 5}.get(n, 0)


def check_tpu_queue(project: str) -> list[dict]:
    """Query all TPU queued resources across known zones."""
    if not project:
        return []
    zones = ["us-east1-d", "europe-west4-a", "europe-west4-b",
             "us-central1-a", "us-central2-b"]
    resources = []
    for zone in zones:
        try:
            # Use shell=True on Windows because gcloud is a .cmd/.bat wrapper
            cmd = f"gcloud compute tpus queued-resources list --project={project} --zone={zone} --format=json"
            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=10, shell=True,
            )
            if result.returncode == 0 and result.stdout.strip():
                items = json.loads(result.stdout)
                for item in items:
                    name = item.get("name", "").split("/")[-1]
                    state = item.get("state", {}).get("state", "UNKNOWN")
                    acc_type = item.get("tpu", {}).get("nodeSpec", [{}])[0].get(
                        "node", {}).get("acceleratorType", "unknown")
                    resources.append({
                        "name": name, "zone": zone,
                        "state": state, "type": acc_type,
                    })
        except (subprocess.TimeoutExpired, Exception):
            pass
    return resources


def build_api_data(log_dir: Path, sync_state: dict) -> dict:
    """Build the /api/data response from log files."""
    results_path = log_dir / "overnight_results.json"
    results = load_results(results_path)
    completed_names = {r["experiment"] for r in results if "experiment" in r}

    experiments = {}
    latest_mtime = 0

    # Scan recursively for JSONL files
    for jsonl_path in sorted(log_dir.rglob("*.jsonl")):
        name = jsonl_path.stem
        header, records = load_jsonl(jsonl_path)
        if not records and not header:
            continue

        mtime = jsonl_path.stat().st_mtime
        latest_mtime = max(latest_mtime, mtime)
        is_running = name not in completed_names and (time.time() - mtime < 300)

        # Extract metadata from header or results
        max_steps = 0
        bits = 0
        block_size = 64
        use_nativebit = True
        batch_size = 0
        context_len = 0

        if header:
            max_steps = header.get("max_steps", 0)
            bits = n_codebook_to_bits(header.get("n_codebook", 8))
            block_size = header.get("block_size", 64)
            use_nativebit = header.get("use_nativebit", True)
            batch_size = header.get("batch_size", 0)
            context_len = header.get("context_len", header.get("seq_len", 0))

        # Try to get from results if header missing
        for r in results:
            if r.get("experiment") == name and "config" in r:
                cfg = r["config"]
                max_steps = max_steps or cfg.get("max_steps", 0)
                bits = bits or n_codebook_to_bits(cfg.get("n_codebook", 8))
                block_size = cfg.get("block_size", block_size)
                use_nativebit = r.get("use_nativebit", use_nativebit)
                batch_size = batch_size or cfg.get("batch_size", 0)
                context_len = context_len or cfg.get("context_len", cfg.get("seq_len", 0))
                break

        latest_step = records[-1]["step"] if records else 0

        # Detect device from name or parent directory
        device = "tpu" if ("tpu" in name.lower() or "tpu" in str(jsonl_path.parent).lower()) else "cuda"

        experiments[name] = {
            "logs": records,
            "is_running": is_running,
            "latest_step": latest_step,
            "max_steps": max_steps,
            "bits": bits,
            "block_size": block_size,
            "use_nativebit": use_nativebit,
            "device": device,
            "batch_size": batch_size,
            "context_len": context_len,
        }

    # Cap experiments to 50 most recent by mtime, limit log points per experiment
    MAX_EXPERIMENTS = 30
    MAX_LOG_POINTS = 200
    if len(experiments) > MAX_EXPERIMENTS:
        by_step = sorted(experiments.items(), key=lambda x: x[1]["latest_step"], reverse=True)
        experiments = dict(by_step[:MAX_EXPERIMENTS])
    for exp in experiments.values():
        if len(exp["logs"]) > MAX_LOG_POINTS:
            # Downsample: keep first, last, and evenly spaced points
            logs = exp["logs"]
            step = max(1, len(logs) // MAX_LOG_POINTS)
            exp["logs"] = logs[::step]
            if exp["logs"][-1] != logs[-1]:
                exp["logs"].append(logs[-1])

    # TPU queue status (from cache, updated by background thread)
    tpu_queue = sync_state.get("_tpu_queue_cache", [])

    # Experiment queue (from queue.json written by overnight script)
    exp_queue = []
    queue_path = log_dir / "queue.json"
    if queue_path.exists():
        try:
            with open(queue_path) as f:
                exp_queue = json.load(f)
        except (json.JSONDecodeError, OSError):
            pass

    return {
        "experiments": experiments,
        "results": results,
        "tpu_queue": tpu_queue,
        "exp_queue": exp_queue,
        "sync": {
            "last_sync_iso": sync_state.get("last_sync_iso", ""),
            "tpu_reachable": sync_state.get("tpu_reachable", False),
            "syncing": sync_state.get("syncing", False),
            "error": sync_state.get("error"),
            "last_log_modified_iso": datetime.fromtimestamp(
                latest_mtime, tz=timezone.utc
            ).isoformat() if latest_mtime > 0 else "",
            "mode": sync_state.get("mode", "local"),
        },
    }


# ---------------------------------------------------------------------------
# TPU sync thread
# ---------------------------------------------------------------------------

class TPUSyncThread(threading.Thread):
    """Background thread that SCPs logs from TPU VM every 30s."""

    def __init__(self, tpu_name: str, tpu_zone: str, tpu_project: str,
                 tpu_log_dir: str, local_dir: Path, interval: int = 30):
        super().__init__(daemon=True)
        self.tpu_name = tpu_name
        self.tpu_zone = tpu_zone
        self.tpu_project = tpu_project
        self.tpu_log_dir = tpu_log_dir
        self.local_dir = local_dir
        self.interval = interval
        self.state = {
            "last_sync_iso": "",
            "tpu_reachable": False,
            "syncing": False,
            "error": None,
            "mode": "tpu",
            "consecutive_failures": 0,
        }
        self._running = True

    def run(self):
        while self._running:
            self._sync()
            time.sleep(self.interval)

    def _sync(self):
        if self.state["syncing"]:
            return  # guard against overlap

        self.state["syncing"] = True
        self.local_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Use SSH to list and copy files (more reliable than glob over SCP)
            # First, get list of files
            list_cmd = (
                f"gcloud compute tpus tpu-vm ssh {self.tpu_name}"
                f" --project={self.tpu_project} --zone={self.tpu_zone}"
                f" --command=\"ls {self.tpu_log_dir}/*.jsonl {self.tpu_log_dir}/overnight_results.json 2>/dev/null\""
            )
            result = subprocess.run(
                list_cmd, timeout=20, capture_output=True, text=True, shell=True
            )

            if result.returncode == 0 and result.stdout.strip():
                files = result.stdout.strip().split("\n")
                for remote_path in files:
                    remote_path = remote_path.strip()
                    if not remote_path:
                        continue
                    fname = os.path.basename(remote_path)
                    scp_cmd = (
                        f"gcloud compute tpus tpu-vm scp"
                        f" {self.tpu_name}:{remote_path}"
                        f" {str(self.local_dir / fname)}"
                        f" --project={self.tpu_project} --zone={self.tpu_zone}"
                    )
                    subprocess.run(scp_cmd, timeout=20, capture_output=True, text=True, shell=True)

            self.state["last_sync_iso"] = datetime.now(timezone.utc).isoformat()
            self.state["tpu_reachable"] = True
            self.state["error"] = None
            self.state["consecutive_failures"] = 0

        except subprocess.TimeoutExpired:
            self.state["consecutive_failures"] += 1
            self.state["error"] = "SCP timeout"
            if self.state["consecutive_failures"] >= 3:
                self.state["tpu_reachable"] = False
        except Exception as e:
            self.state["consecutive_failures"] += 1
            self.state["error"] = str(e)[:100]
            if self.state["consecutive_failures"] >= 3:
                self.state["tpu_reachable"] = False
        finally:
            self.state["syncing"] = False

    def stop(self):
        self._running = False


# ---------------------------------------------------------------------------
# HTTP server
# ---------------------------------------------------------------------------

def make_handler(log_dir: Path, sync_state: dict):
    """Create an HTTP request handler with access to log_dir and sync state."""

    class DashboardHandler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self._serve_html()
            elif self.path == "/api/data":
                self._serve_api()
            elif self.path == "/api/sync":
                sync_state["force_sync"] = True
                self._json_response({"ok": True})
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            else:
                self.send_error(404)

        def _serve_html(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(DASHBOARD_HTML.encode("utf-8"))

        def _serve_api(self):
            data = build_api_data(log_dir, sync_state)
            self._json_response(data)

        def _json_response(self, obj):
            body = json.dumps(obj).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format, *args):
            pass  # silence request logs

    return DashboardHandler


# ---------------------------------------------------------------------------
# HTML Dashboard (embedded)
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>NativeBit TPU Dashboard</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
:root {
  --bg: #fafaf9; --bg-card: #ffffff; --border: #e7e5e4;
  --text: #1e293b; --text2: #64748b; --text3: #94a3b8;
  --emerald: #059669; --orange: #f97316; --red: #ef4444; --amber: #f59e0b;
  --blue: #3b82f6;
}
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
       background: var(--bg); color: var(--text); font-size: 14px;
       overflow-x: hidden; }
.mono { font-family: 'SF Mono', 'Cascadia Code', 'Consolas', monospace; }
.header { padding: 16px 24px; border-bottom: 1px solid var(--border);
           display: flex; align-items: center; gap: 16px; background: var(--bg-card); }
.header h1 { font-size: 18px; font-weight: 600; }
.badge { font-size: 11px; padding: 2px 8px; border-radius: 9999px; font-weight: 500; }
.badge-nb { background: #d1fae5; color: #065f46; }
.badge-fp { background: #ffedd5; color: #9a3412; }
.tabs { display: flex; gap: 0; border-bottom: 1px solid var(--border);
        background: var(--bg-card); padding: 0 24px; }
.tab { padding: 10px 20px; cursor: pointer; border-bottom: 2px solid transparent;
       color: var(--text2); font-weight: 500; font-size: 13px; }
.tab:hover { color: var(--text); }
.tab.active { color: var(--emerald); border-bottom-color: var(--emerald); }
.content { padding: 20px 24px; max-width: 1400px; margin: 0 auto; }
.tab-panel { display: none; }
.tab-panel.active { display: block; }
.card { background: var(--bg-card); border: 1px solid var(--border);
        border-radius: 8px; padding: 16px; margin-bottom: 16px;
        max-width: 100%; overflow: hidden; }
.card h3 { font-size: 13px; color: var(--text2); font-weight: 500;
            margin-bottom: 8px; text-transform: uppercase; letter-spacing: 0.5px; }
.grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid-4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.stat { text-align: center; }
.stat .value { font-size: 28px; font-weight: 700; line-height: 1.1; }
.stat .label { font-size: 11px; color: var(--text2); margin-top: 2px;
               text-transform: uppercase; letter-spacing: 0.5px; }
.progress-bar { height: 6px; background: #e2e8f0; border-radius: 3px;
                overflow: hidden; margin: 8px 0; }
.progress-fill { height: 100%; background: var(--emerald); border-radius: 3px;
                 transition: width 0.5s ease; }
.live-top { display: flex; justify-content: space-between; align-items: center;
            flex-wrap: wrap; gap: 12px; margin-bottom: 8px; }
.live-top .name { font-size: 16px; font-weight: 600; }
.live-top .desc { font-size: 12px; color: var(--text2); }
.status-dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block;
              margin-right: 6px; }
.status-running { background: var(--emerald); animation: pulse 1.5s infinite; }
.status-stalled { background: var(--amber); }
.status-failed { background: var(--red); }
.status-idle { background: var(--text3); }
@keyframes pulse { 0%,100% { opacity: 1; } 50% { opacity: 0.4; } }
.banner { padding: 10px 16px; border-radius: 6px; margin-bottom: 12px;
          font-size: 13px; font-weight: 500; }
.banner-amber { background: #fffbeb; border: 1px solid #fde68a; color: #92400e; }
.banner-red { background: #fef2f2; border: 1px solid #fecaca; color: #991b1b; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; padding: 8px 12px; border-bottom: 2px solid var(--border);
     color: var(--text2); font-weight: 600; font-size: 11px;
     text-transform: uppercase; letter-spacing: 0.5px; cursor: pointer; }
th:hover { color: var(--text); }
td { padding: 8px 12px; border-bottom: 1px solid var(--border); }
tr:hover td { background: #f8fafc; }
.rank { font-weight: 700; color: var(--text2); width: 40px; }
.filter-bar { display: flex; gap: 12px; margin-bottom: 16px; align-items: center; flex-wrap: wrap; }
.filter-bar label { font-size: 12px; color: var(--text2); font-weight: 500; }
.filter-bar select { font-size: 12px; padding: 4px 8px; border: 1px solid var(--border);
                     border-radius: 4px; background: var(--bg-card); }
.footer { position: fixed; bottom: 0; left: 0; right: 0; padding: 8px 24px;
          background: var(--bg-card); border-top: 1px solid var(--border);
          display: flex; gap: 24px; font-size: 11px; color: var(--text2); z-index: 10; }
.footer .sep { color: var(--border); }
.chart-wrap { overflow: hidden; max-width: 100%; }
.table-wrap { overflow-x: auto; max-width: 100%; -webkit-overflow-scrolling: touch; }
@media (max-width: 900px) {
  .grid-2 { grid-template-columns: 1fr; }
  .grid-4 { grid-template-columns: 1fr 1fr; }
}
</style>
</head>
<body>
<div class="header">
  <h1>NativeBit</h1>
  <span style="color:var(--text2)">TPU Training Dashboard</span>
</div>
<div class="tabs" id="tab-bar"></div>
<div id="tpu-queue-bar" style="padding:8px 24px;background:#f1f5f9;border-bottom:1px solid var(--border);font-size:12px;display:none;overflow-x:auto;max-width:100%">
  <span style="font-weight:600;color:var(--text2);margin-right:12px">TPU QUEUE</span>
  <span id="tpu-queue-items"></span>
</div>
<div class="content" style="padding-bottom: 50px;">

<div id="tab-live" class="tab-panel active">
  <div id="live-banner"></div>
  <div class="card">
    <div class="live-top">
      <div><span id="live-status-dot" class="status-dot status-idle"></span><span id="live-name" class="name">Waiting...</span>
        <span id="live-desc" class="desc"></span></div>
      <div style="text-align:right"><span id="live-progress-text" class="mono" style="font-size:13px"></span></div>
    </div>
    <div class="progress-bar"><div id="live-progress-fill" class="progress-fill" style="width:0%"></div></div>
    <div style="display:flex;justify-content:space-between;font-size:12px;color:var(--text2)">
      <span id="live-elapsed"></span><span id="live-eta"></span>
    </div>
  </div>
  <div class="grid-4" style="margin-bottom:16px">
    <div class="card stat"><div id="live-loss" class="value mono">--</div><div class="label">Loss</div></div>
    <div class="card stat"><div id="live-ppl" class="value mono">--</div><div class="label">Perplexity</div></div>
    <div class="card stat"><div id="live-dead" class="value mono">--</div><div class="label">Dead %</div></div>
    <div class="card stat"><div id="live-speed" class="value mono">--</div><div class="label">Tokens/sec</div></div>
  </div>
  <div class="card" style="margin-bottom:16px;padding:8px;border-bottom:2px solid var(--border)">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">
      <span style="font-size:11px;color:var(--text2);font-weight:500;text-transform:uppercase;letter-spacing:0.5px">Throughput</span>
      <span id="live-throughput-val" class="mono" style="font-size:11px;color:var(--text2)"></span>
    </div>
    <div class="chart-wrap"><div id="chart-live-throughput" style="height:48px"></div></div>
  </div>
  <div class="grid-2">
    <div class="card"><h3>Loss &amp; Perplexity</h3><div class="chart-wrap"><div id="chart-live-loss" style="height:300px"></div></div></div>
    <div class="card"><h3>Codebook Health</h3><div class="chart-wrap"><div id="chart-live-dead" style="height:300px"></div></div></div>
    <div class="card"><h3>Gradient Ratio (CB/Weight)</h3><div class="chart-wrap"><div id="chart-live-grad" style="height:300px"></div></div></div>
    <div class="card"><h3>Learning Rate</h3><div class="chart-wrap"><div id="chart-live-lr" style="height:300px"></div></div></div>
  </div>
</div>

<div id="tab-compare" class="tab-panel">
  <div class="filter-bar">
    <label>Device:</label><select id="filter-device" onchange="updateCompare()">
      <option value="all">All</option><option value="tpu">TPU</option><option value="cuda">CUDA</option></select>
    <label>Type:</label><select id="filter-type" onchange="updateCompare()">
      <option value="all">All</option><option value="nb">NativeBit</option><option value="fp">Float</option></select>
    <label>Bits:</label><select id="filter-bits" onchange="updateCompare()">
      <option value="all">All</option><option value="2">2-bit</option><option value="3">3-bit</option><option value="4">4-bit</option></select>
    <label>Block Size:</label><select id="filter-bs" onchange="updateCompare()">
      <option value="all">All</option><option value="64">64</option><option value="128">128</option><option value="256">256</option></select>
    <label>Max Steps:</label><select id="filter-maxsteps" onchange="updateCompare()">
      <option value="all">All</option><option value="lt1000">&lt; 1000</option><option value="1000-5000">1000-5000</option><option value="5000-10000">5000-10000</option><option value="10000+">10000+</option></select>
  </div>
  <div class="grid-2">
    <div class="card"><h3>Loss</h3><div class="chart-wrap"><div id="chart-cmp-loss" style="height:350px"></div></div></div>
    <div class="card"><h3>Perplexity</h3><div class="chart-wrap"><div id="chart-cmp-ppl" style="height:350px"></div></div></div>
    <div class="card"><h3>Dead Entry %</h3><div class="chart-wrap"><div id="chart-cmp-dead" style="height:350px"></div></div></div>
    <div class="card"><h3>Gradient Ratio</h3><div class="chart-wrap"><div id="chart-cmp-grad" style="height:350px"></div></div></div>
  </div>
  <div class="card"><h3>Comparison Table</h3><div class="table-wrap">
    <table id="compare-table"><thead><tr>
      <th onclick="sortTable('compare-table',0)">Name</th><th onclick="sortTable('compare-table',1)">Type</th>
      <th onclick="sortTable('compare-table',2)">Bits</th><th onclick="sortTable('compare-table',3)">Block</th>
      <th onclick="sortTable('compare-table',4)">Test PPL</th><th onclick="sortTable('compare-table',5)">Val PPL</th>
      <th onclick="sortTable('compare-table',6)">Dead%</th><th onclick="sortTable('compare-table',7)">Time</th>
    </tr></thead><tbody></tbody></table>
  </div></div>
</div>

<div id="tab-queue" class="tab-panel">
  <div class="card"><h3>Experiment Queue</h3><div class="table-wrap">
    <table id="exp-queue-table"><thead><tr>
      <th style="width:30px">#</th><th>Experiment</th><th>Description</th>
      <th>Type</th><th>Status</th>
    </tr></thead><tbody id="exp-queue-list"></tbody></table>
  </div></div>
</div>

<div id="tab-leaderboard" class="tab-panel">
  <div class="grid-4" id="leader-summary" style="margin-bottom:16px"></div>
  <div class="card"><h3>Results Ranked by Test PPL</h3><div class="table-wrap">
    <table id="leader-table"><thead><tr>
      <th style="width:40px">#</th><th onclick="sortTable('leader-table',1)">Experiment</th>
      <th onclick="sortTable('leader-table',2)">Type</th><th onclick="sortTable('leader-table',3)">Bits</th>
      <th onclick="sortTable('leader-table',4)">Block</th><th onclick="sortTable('leader-table',5)">Test PPL</th>
      <th onclick="sortTable('leader-table',6)">Val PPL</th><th onclick="sortTable('leader-table',7)">BPB</th>
      <th onclick="sortTable('leader-table',8)">Time</th><th>Status</th>
    </tr></thead><tbody></tbody></table>
  </div></div>
</div>

</div>
<div class="footer">
  <span id="footer-sync">Sync: --</span><span class="sep">|</span>
  <span id="footer-log">Last log: --</span><span class="sep">|</span>
  <span id="footer-tpu">TPU: --</span><span class="sep">|</span>
  <span id="footer-mode">Mode: --</span>
</div>

<script>
var DATA = null;
var COLORS = ['#059669','#f97316','#3b82f6','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f','#bcbd22','#17becf'];
var LB = {paper_bgcolor:'transparent',plot_bgcolor:'transparent',
  font:{family:'-apple-system,sans-serif',size:11,color:'#64748b'},
  margin:{l:50,r:20,t:10,b:40},xaxis:{gridcolor:'#e7e5e4',title:'Step'},
  yaxis:{gridcolor:'#e7e5e4'},legend:{orientation:'h',y:-0.2},hovermode:'x unified'};

// Tabs
var tabNames = ['live','compare','queue','leaderboard'];
var tabBar = document.getElementById('tab-bar');
tabNames.forEach(function(n) {
  var el = document.createElement('div');
  el.className = 'tab' + (n === 'live' ? ' active' : '');
  el.textContent = n.charAt(0).toUpperCase() + n.slice(1);
  el.setAttribute('data-tab', n);
  el.onclick = function() { switchTab(n); };
  tabBar.appendChild(el);
});

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(function(t) { t.classList.toggle('active', t.getAttribute('data-tab') === name); });
  document.querySelectorAll('.tab-panel').forEach(function(p) { p.classList.toggle('active', p.id === 'tab-' + name); });
  if (DATA) render(DATA);
}

function n2bits(n) { return {4:2,8:3,16:4,32:5}[n] || 0; }
function fmtTime(s) {
  if (s < 60) return Math.round(s) + 's';
  if (s < 3600) return Math.round(s / 60) + 'm';
  return Math.floor(s / 3600) + 'h ' + Math.round((s % 3600) / 60) + 'm';
}
function esc(s) { var d = document.createElement('div'); d.appendChild(document.createTextNode(s)); return d.innerHTML; }

async function fetchData() {
  try { var r = await fetch('/api/data'); DATA = await r.json(); render(DATA); }
  catch(e) { console.error('Fetch failed:', e); }
}

function render(d) { renderTPUQueue(d); renderLive(d); renderCompare(d); renderQueue(d); renderLeaderboard(d); renderFooter(d); }

function renderLive(d) {
  var running = null, runName = '';
  var entries = Object.entries(d.experiments);
  for (var i = 0; i < entries.length; i++) {
    if (entries[i][1].is_running) { running = entries[i][1]; runName = entries[i][0]; break; }
  }
  if (!running) {
    var best = 0;
    for (var i = 0; i < entries.length; i++) {
      if (entries[i][1].latest_step > best) { best = entries[i][1].latest_step; running = entries[i][1]; runName = entries[i][0]; }
    }
  }
  if (!running || !running.logs.length) {
    document.getElementById('live-name').textContent = 'No active experiments -- showing most recent';
    document.getElementById('live-status-dot').className = 'status-dot status-idle';
    return;
  }
  var logs = running.logs, last = logs[logs.length - 1];
  var maxSteps = running.max_steps || last.step || 1;
  var pct = Math.min(100, (last.step / maxSteps) * 100);

  var dot = document.getElementById('live-status-dot');
  dot.className = 'status-dot ' + (running.is_running ? 'status-running' : 'status-idle');
  document.getElementById('live-name').textContent = runName;
  document.getElementById('live-desc').textContent =
    (running.use_nativebit ? 'NativeBit ' + running.bits + '-bit' : 'Float') + ' | bs=' + running.block_size;
  document.getElementById('live-progress-text').textContent =
    'Step ' + last.step.toLocaleString() + ' / ' + maxSteps.toLocaleString();
  document.getElementById('live-progress-fill').style.width = pct + '%';

  var elapsed = last.elapsed_s || 0;
  document.getElementById('live-elapsed').textContent = 'Running: ' + fmtTime(elapsed);
  if (last.step > 0 && running.is_running) {
    var eta = (elapsed / last.step) * (maxSteps - last.step);
    document.getElementById('live-eta').textContent = 'ETA: ~' + fmtTime(eta);
  } else {
    document.getElementById('live-eta').textContent = running.is_running ? '' : 'Completed';
  }

  document.getElementById('live-loss').textContent = last.loss != null ? last.loss.toFixed(3) : '--';
  document.getElementById('live-ppl').textContent = last.perplexity != null ? last.perplexity.toFixed(1) : '--';
  document.getElementById('live-dead').textContent = last.dead_pct != null ? last.dead_pct.toFixed(1) + '%' : '--';

  // Tokens/sec computation — prefer experiment-level metadata, fall back to log entries
  var batchSize = running.batch_size || 0;
  var contextLen = running.context_len || 0;
  if ((!batchSize || !contextLen) && running.logs.length > 0) {
    var firstLog = running.logs[0];
    batchSize = batchSize || firstLog.batch_size || 0;
    contextLen = contextLen || firstLog.context_len || 0;
  }

  if (logs.length >= 2) {
    var prev = logs[logs.length - 2];
    var dt = (last.elapsed_s - prev.elapsed_s) || 1;
    var ds = last.step - prev.step;
    var stepsPerSec = ds / dt;
    if (batchSize > 0 && contextLen > 0) {
      var tokPerSec = stepsPerSec * batchSize * contextLen;
      document.getElementById('live-speed').textContent = tokPerSec >= 1000 ? (tokPerSec / 1000).toFixed(1) + 'k' : Math.round(tokPerSec);
    } else {
      // Fallback to steps/sec if batch info unavailable
      document.getElementById('live-speed').textContent = stepsPerSec.toFixed(1) + ' st/s';
    }
  }

  // Throughput sparkline (last 20 entries)
  var throughputSteps = [], throughputVals = [];
  var sparkLogs = logs.slice(-21); // need 21 to get 20 deltas
  for (var si = 1; si < sparkLogs.length; si++) {
    var sDt = (sparkLogs[si].elapsed_s - sparkLogs[si - 1].elapsed_s) || 1;
    var sDs = sparkLogs[si].step - sparkLogs[si - 1].step;
    var sps = sDs / sDt;
    throughputSteps.push(sparkLogs[si].step);
    if (batchSize > 0 && contextLen > 0) {
      throughputVals.push(sps * batchSize * contextLen);
    } else {
      throughputVals.push(sps);
    }
  }
  if (throughputVals.length > 0) {
    var lastTp = throughputVals[throughputVals.length - 1];
    var tpLabel = (batchSize > 0 && contextLen > 0)
      ? (lastTp >= 1000 ? (lastTp / 1000).toFixed(1) + 'k tok/s' : Math.round(lastTp) + ' tok/s')
      : lastTp.toFixed(1) + ' st/s';
    document.getElementById('live-throughput-val').textContent = tpLabel;
  }
  if (throughputVals.length > 1) {
    Plotly.react('chart-live-throughput',
      [{x: throughputSteps, y: throughputVals, mode: 'lines', line: {color: COLORS[0], width: 2},
        fill: 'tozeroy', fillcolor: 'rgba(5,150,105,0.08)', hovertemplate: '%{y:.0f}<extra></extra>'}],
      {paper_bgcolor: 'transparent', plot_bgcolor: 'transparent',
       margin: {l: 0, r: 0, t: 0, b: 0},
       xaxis: {visible: false}, yaxis: {visible: false},
       hovermode: 'x', showlegend: false},
      {responsive: true, displayModeBar: false});
  }

  // Banner
  var bannerEl = document.getElementById('live-banner');
  bannerEl.textContent = '';
  if (d.sync.last_log_modified_iso && running.is_running) {
    var logAge = (Date.now() - new Date(d.sync.last_log_modified_iso).getTime()) / 1000;
    if (logAge > 300) {
      var b = document.createElement('div');
      b.className = 'banner banner-amber';
      b.textContent = 'Training appears stalled -- no log update for ' + fmtTime(logAge);
      bannerEl.appendChild(b);
    }
  }
  for (var i = 0; i < d.results.length; i++) {
    if (d.results[i].error) {
      var b = document.createElement('div');
      b.className = 'banner banner-red';
      b.textContent = d.results[i].experiment + ' failed: ' + d.results[i].error;
      bannerEl.appendChild(b);
    }
  }

  // Charts
  var steps = logs.map(function(r){return r.step;});
  Plotly.react('chart-live-loss',
    [{x:steps, y:logs.map(function(r){return r.loss;}), name:'Loss', line:{color:COLORS[0]}},
     {x:steps, y:logs.map(function(r){return r.perplexity;}), name:'PPL', line:{color:COLORS[1]}, yaxis:'y2'}],
    Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Loss'},
      yaxis2:{overlaying:'y',side:'right',title:'PPL',type:'log',gridcolor:'transparent'}}),
    {responsive:true, displayModeBar:false});

  if (logs[0] && logs[0].dead_pct != null) {
    Plotly.react('chart-live-dead',
      [{x:steps, y:logs.map(function(r){return r.dead_pct||0;}), fill:'tozeroy',
        fillcolor:'rgba(239,68,68,0.1)', line:{color:COLORS[3]}, name:'Dead %'}],
      Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Dead %'}}),
      {responsive:true, displayModeBar:false});
  }

  if (logs[0] && logs[0].grad_ratio_cb_w != null) {
    Plotly.react('chart-live-grad',
      [{x:steps, y:logs.map(function(r){return r.grad_ratio_cb_w||0;}), line:{color:COLORS[2]}, name:'CB/W Ratio'},
       {x:[steps[0],steps[steps.length-1]], y:[1,1], mode:'lines', line:{color:'#94a3b8',dash:'dash'}, showlegend:false}],
      Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Gradient Ratio'}}),
      {responsive:true, displayModeBar:false});
  }

  Plotly.react('chart-live-lr',
    [{x:steps, y:logs.map(function(r){return r.lr;}), line:{color:'#1e293b'}, name:'LR'}],
    Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Learning Rate'}}),
    {responsive:true, displayModeBar:false});
}

function updateCompare() { if (DATA) renderCompare(DATA); }

function renderCompare(d) {
  var fDev = document.getElementById('filter-device').value;
  var fType = document.getElementById('filter-type').value;
  var fBits = document.getElementById('filter-bits').value;
  var fBs = document.getElementById('filter-bs').value;
  var fMaxSteps = document.getElementById('filter-maxsteps').value;

  var lossT = [], pplT = [], deadT = [], gradT = [], ci = 0;
  var entries = Object.entries(d.experiments);
  for (var i = 0; i < entries.length; i++) {
    var name = entries[i][0], exp = entries[i][1];
    if (fDev !== 'all' && (exp.device || 'cuda') !== fDev) continue;
    if (fType === 'nb' && !exp.use_nativebit) continue;
    if (fType === 'fp' && exp.use_nativebit) continue;
    if (fBits !== 'all' && exp.bits !== parseInt(fBits)) continue;
    if (fBs !== 'all' && exp.block_size !== parseInt(fBs)) continue;
    if (fMaxSteps !== 'all') {
      var ls = exp.latest_step || 0;
      if (fMaxSteps === 'lt1000' && ls >= 1000) continue;
      if (fMaxSteps === '1000-5000' && (ls < 1000 || ls > 5000)) continue;
      if (fMaxSteps === '5000-10000' && (ls < 5000 || ls > 10000)) continue;
      if (fMaxSteps === '10000+' && ls < 10000) continue;
    }
    var steps = exp.logs.map(function(r){return r.step;});
    var c = COLORS[ci % COLORS.length]; ci++;
    lossT.push({x:steps, y:exp.logs.map(function(r){return r.loss;}), name:name, line:{color:c}});
    pplT.push({x:steps, y:exp.logs.map(function(r){return r.perplexity;}), name:name, line:{color:c}});
    if (exp.logs[0] && exp.logs[0].dead_pct != null)
      deadT.push({x:steps, y:exp.logs.map(function(r){return r.dead_pct||0;}), name:name, line:{color:c}});
    if (exp.logs[0] && exp.logs[0].grad_ratio_cb_w != null)
      gradT.push({x:steps, y:exp.logs.map(function(r){return r.grad_ratio_cb_w||0;}), name:name, line:{color:c}});
  }

  Plotly.react('chart-cmp-loss', lossT, Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Loss'}}), {responsive:true,displayModeBar:false});
  Plotly.react('chart-cmp-ppl', pplT, Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'PPL',type:'log'}}), {responsive:true,displayModeBar:false});
  Plotly.react('chart-cmp-dead', deadT, Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Dead %'}}), {responsive:true,displayModeBar:false});
  Plotly.react('chart-cmp-grad', gradT, Object.assign({}, LB, {yaxis:{gridcolor:'#e7e5e4',title:'Ratio'}}), {responsive:true,displayModeBar:false});

  // Table
  var tbody = document.querySelector('#compare-table tbody');
  tbody.textContent = '';
  for (var i = 0; i < d.results.length; i++) {
    var r = d.results[i]; if (r.error) continue;
    var cfg = r.config || {}, bits = n2bits(cfg.n_codebook || 8), isNb = r.use_nativebit;
    var rDev = (r.experiment && r.experiment.toLowerCase().indexOf('tpu') >= 0) ? 'tpu' : 'cuda';
    if (fDev !== 'all' && rDev !== fDev) continue;
    if (fType === 'nb' && !isNb) continue;
    if (fType === 'fp' && isNb) continue;
    if (fBits !== 'all' && bits !== parseInt(fBits)) continue;
    if (fBs !== 'all' && (cfg.block_size||64) !== parseInt(fBs)) continue;
    var tr = document.createElement('tr');
    var cells = [
      r.experiment, isNb ? 'NB' : 'FP', isNb ? bits + '-bit' : '-',
      cfg.block_size || '-', r.test_ppl ? r.test_ppl.toFixed(2) : '-',
      r.val_ppl ? r.val_ppl.toFixed(2) : '-', '-',
      r.elapsed_min ? r.elapsed_min.toFixed(0) + 'm' : '-'
    ];
    cells.forEach(function(txt, ci) {
      var td = document.createElement('td');
      td.textContent = txt;
      if (ci >= 2) td.className = 'mono';
      if (ci === 1) {
        td.textContent = '';
        var sp = document.createElement('span');
        sp.className = 'badge ' + (isNb ? 'badge-nb' : 'badge-fp');
        sp.textContent = txt;
        td.appendChild(sp);
      }
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  }
}

function renderQueue(d) {
  var tbody = document.getElementById('exp-queue-list');
  if (!d.exp_queue || !d.exp_queue.length) { tbody.textContent = ''; return; }
  tbody.textContent = '';
  var statusColors = {completed:'var(--emerald)',running:'var(--blue)',pending:'var(--text3)',failed:'var(--red)'};
  d.exp_queue.forEach(function(q, idx) {
    var tr = document.createElement('tr');
    // Number
    var tdNum = document.createElement('td');
    tdNum.className = 'rank';
    tdNum.textContent = '' + (idx + 1);
    tr.appendChild(tdNum);
    // Name
    var tdName = document.createElement('td');
    tdName.style.fontWeight = '600';
    tdName.textContent = q.name;
    tr.appendChild(tdName);
    // Description
    var tdDesc = document.createElement('td');
    tdDesc.style.color = 'var(--text2)';
    tdDesc.textContent = q.description;
    tr.appendChild(tdDesc);
    // Type
    var tdType = document.createElement('td');
    var badge = document.createElement('span');
    badge.className = 'badge ' + (q.use_nativebit ? 'badge-nb' : 'badge-fp');
    badge.textContent = q.use_nativebit ? 'NB' : 'FP';
    tdType.appendChild(badge);
    tr.appendChild(tdType);
    // Status
    var tdStatus = document.createElement('td');
    tdStatus.style.cssText = 'font-weight:700;font-size:11px;text-transform:uppercase;letter-spacing:0.5px;color:' + (statusColors[q.status] || 'var(--text3)');
    tdStatus.textContent = q.status;
    if (q.status === 'running') tdStatus.style.animation = 'pulse 1.5s infinite';
    tr.appendChild(tdStatus);
    tbody.appendChild(tr);
  });
}

function renderLeaderboard(d) {
  var summary = document.getElementById('leader-summary');
  var valid = d.results.filter(function(r){return !r.error && r.test_ppl;});
  var nbR = valid.filter(function(r){return r.use_nativebit;});
  var fpR = valid.filter(function(r){return !r.use_nativebit;});
  var bestNb = nbR.length ? Math.min.apply(null, nbR.map(function(r){return r.test_ppl;})) : null;
  var bestFp = fpR.length ? Math.min.apply(null, fpR.map(function(r){return r.test_ppl;})) : null;
  var nRunning = Object.keys(d.experiments).length - valid.length;

  summary.textContent = '';
  var cards = [
    {val: bestNb ? bestNb.toFixed(1) : '--', label: 'Best NativeBit PPL', color: 'var(--emerald)'},
    {val: bestFp ? bestFp.toFixed(1) : '--', label: 'Best Float PPL', color: 'var(--orange)'},
    {val: '' + valid.length, label: 'Experiments Done', color: ''},
    {val: '' + nRunning, label: 'Running / Queued', color: ''}
  ];
  cards.forEach(function(c) {
    var card = document.createElement('div');
    card.className = 'card stat';
    var vEl = document.createElement('div');
    vEl.className = 'value mono';
    vEl.textContent = c.val;
    if (c.color) vEl.style.color = c.color;
    var lEl = document.createElement('div');
    lEl.className = 'label';
    lEl.textContent = c.label;
    card.appendChild(vEl);
    card.appendChild(lEl);
    summary.appendChild(card);
  });

  var sorted = d.results.slice().sort(function(a,b){return (a.test_ppl||9999)-(b.test_ppl||9999);});
  var tbody = document.querySelector('#leader-table tbody');
  tbody.textContent = '';
  sorted.forEach(function(r, idx) {
    var cfg = r.config || {}, bits = n2bits(cfg.n_codebook || 8), isNb = r.use_nativebit;
    var failed = !!r.error;
    var tr = document.createElement('tr');
    var vals = [
      failed ? '-' : '' + (idx + 1), r.experiment, isNb ? 'NB' : 'FP',
      isNb ? bits + '-bit' : '-', cfg.block_size || '-',
      r.test_ppl ? r.test_ppl.toFixed(2) : '-',
      r.val_ppl ? r.val_ppl.toFixed(2) : '-',
      r.val_bpb ? r.val_bpb.toFixed(3) : '-',
      r.elapsed_min ? r.elapsed_min.toFixed(0) + 'm' : '-',
      failed ? 'Failed' : 'Done'
    ];
    vals.forEach(function(txt, ci) {
      var td = document.createElement('td');
      td.textContent = txt;
      if (ci === 0) td.className = 'rank';
      else if (ci >= 3 && ci <= 8) td.className = 'mono';
      else if (ci === 5) { td.className = 'mono'; td.style.fontWeight = '700'; }
      if (ci === 2) {
        td.textContent = '';
        var sp = document.createElement('span');
        sp.className = 'badge ' + (isNb ? 'badge-nb' : 'badge-fp');
        sp.textContent = txt;
        td.appendChild(sp);
      }
      if (ci === 9) td.style.color = failed ? 'var(--red)' : 'var(--emerald)';
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
}

function renderTPUQueue(d) {
  var bar = document.getElementById('tpu-queue-bar');
  var items = document.getElementById('tpu-queue-items');
  if (!d.tpu_queue || !d.tpu_queue.length) { bar.style.display = 'none'; return; }
  bar.style.display = 'block';
  items.textContent = '';
  var stateColors = {ACTIVE:'var(--emerald)',PROVISIONING:'var(--blue)',WAITING_FOR_RESOURCES:'var(--amber)',
    PREEMPTED:'var(--red)',FAILED:'var(--red)',SUSPENDING:'var(--amber)',SUSPENDED:'var(--text3)'};
  // Find currently running experiment name
  var runningExpName = '';
  if (d.experiments) {
    var expEntries = Object.entries(d.experiments);
    for (var i = 0; i < expEntries.length; i++) {
      if (expEntries[i][1].is_running) { runningExpName = expEntries[i][0]; break; }
    }
  }
  d.tpu_queue.forEach(function(t) {
    var sp = document.createElement('span');
    sp.style.cssText = 'margin-right:16px;';
    var dot = document.createElement('span');
    dot.style.cssText = 'display:inline-block;width:6px;height:6px;border-radius:50%;margin-right:4px;background:' + (stateColors[t.state] || 'var(--text3)');
    if (t.state === 'ACTIVE') dot.style.animation = 'pulse 1.5s infinite';
    if (t.state === 'PROVISIONING') dot.style.animation = 'pulse 1s infinite';
    sp.appendChild(dot);
    var txt = document.createTextNode(t.name + ' (' + t.type + ', ' + t.zone.split('-').slice(0,2).join('-') + ') ');
    sp.appendChild(txt);
    var badge = document.createElement('span');
    badge.style.cssText = 'font-size:10px;padding:1px 6px;border-radius:4px;font-weight:600;color:white;background:' + (stateColors[t.state] || '#94a3b8');
    badge.textContent = t.state;
    sp.appendChild(badge);
    if (t.state === 'ACTIVE' && runningExpName) {
      var runLabel = document.createTextNode(' \u2014 running: ' + runningExpName);
      sp.appendChild(runLabel);
    }
    items.appendChild(sp);
  });
}

function renderFooter(d) {
  var s = d.sync;
  document.getElementById('footer-mode').textContent = 'Mode: ' + (s.mode || 'local');
  if (s.last_sync_iso) {
    var ago = Math.round((Date.now() - new Date(s.last_sync_iso).getTime()) / 1000);
    document.getElementById('footer-sync').textContent = 'Synced ' + ago + 's ago';
  }
  if (s.last_log_modified_iso) {
    var ago = Math.round((Date.now() - new Date(s.last_log_modified_iso).getTime()) / 1000);
    document.getElementById('footer-log').textContent = 'Last log: ' + ago + 's ago';
  }
  document.getElementById('footer-tpu').textContent =
    s.tpu_reachable ? 'TPU: connected' :
    s.mode === 'tpu' ? 'TPU: ' + (s.error || 'connecting...') : 'TPU: n/a';
}

function sortTable(id, col) {
  var table = document.getElementById(id);
  var rows = Array.from(table.querySelectorAll('tbody tr'));
  var dir = table.dataset.sortDir === 'asc' ? 'desc' : 'asc';
  table.dataset.sortDir = dir;
  rows.sort(function(a, b) {
    var va = a.cells[col].textContent.trim(), vb = b.cells[col].textContent.trim();
    var na = parseFloat(va), nb = parseFloat(vb);
    if (!isNaN(na) && !isNaN(nb)) return dir === 'asc' ? na - nb : nb - na;
    return dir === 'asc' ? va.localeCompare(vb) : vb.localeCompare(va);
  });
  var tbody = table.querySelector('tbody');
  rows.forEach(function(r) { tbody.appendChild(r); });
}

fetchData();
setInterval(fetchData, 10000);
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="NativeBit TPU Training Dashboard")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--log-dir", type=str, default=None,
                        help="Local log directory (local-only mode)")
    # TPU sync options
    parser.add_argument("--tpu-name", type=str, default=None)
    parser.add_argument("--tpu-zone", type=str, default=None)
    parser.add_argument("--tpu-project", type=str, default=None)
    parser.add_argument("--tpu-log-dir", type=str, default="~/NativeBit/logs/tpu")
    parser.add_argument("--sync-dir", type=str, default="logs/tpu-sync")
    parser.add_argument("--sync-interval", type=int, default=30)
    parser.add_argument("--gcp-project", type=str, default=None,
                        help="GCP project ID for TPU queue status (works in local mode too)")
    args = parser.parse_args()

    # Determine mode
    sync_state = {"mode": "local", "tpu_reachable": False,
                  "tpu_project": args.gcp_project or args.tpu_project or ""}
    sync_thread = None

    if args.tpu_name and args.tpu_zone and args.tpu_project:
        # TPU sync mode
        try:
            subprocess.run(["gcloud", "--version"], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            print("ERROR: gcloud CLI not found. Install it or use --log-dir for local mode.")
            sys.exit(1)

        log_dir = Path(args.sync_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        sync_thread = TPUSyncThread(
            args.tpu_name, args.tpu_zone, args.tpu_project,
            args.tpu_log_dir, log_dir, args.sync_interval,
        )
        sync_state = sync_thread.state
        sync_thread.start()
        print(f"TPU sync: {args.tpu_name} ({args.tpu_zone}) -> {log_dir}/")
    elif args.log_dir:
        log_dir = Path(args.log_dir)
    else:
        log_dir = Path("logs")

    if not log_dir.exists():
        log_dir.mkdir(parents=True, exist_ok=True)

    # Background TPU queue poller (non-blocking)
    tpu_project = sync_state.get("tpu_project", "")
    if tpu_project:
        def _poll_tpu_queue():
            while True:
                try:
                    sync_state["_tpu_queue_cache"] = check_tpu_queue(tpu_project)
                except Exception:
                    pass
                time.sleep(30)
        t = threading.Thread(target=_poll_tpu_queue, daemon=True)
        t.start()

    handler = make_handler(log_dir, sync_state)
    server = HTTPServer(("0.0.0.0", args.port), handler)
    print(f"Dashboard: http://localhost:{args.port}")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        if sync_thread:
            sync_thread.stop()
        server.shutdown()


if __name__ == "__main__":
    main()
