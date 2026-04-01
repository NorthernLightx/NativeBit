"""Clean 350M comparison: float vs NativeBit 3-bit.

Runs each in a separate subprocess to avoid TPU memory conflicts.
Uses jitted requantize (no memory leak), rq=200, checkpointing.
"""
import subprocess, sys, os, json

CONFIGS = [
    {"nativebit": False, "name": "350m_clean_float"},
    {"nativebit": True,  "name": "350m_clean_nb3"},
]

os.makedirs("logs/jax", exist_ok=True)
results = {}

for cfg in CONFIGS:
    label = "NB 3-bit" if cfg["nativebit"] else "Float"
    print(f"\n=== 350M {label} (v6e-8, bs=8) ===")

    code = f"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nativebit_jax.train import train
from configs.tpu import TPULargeConfig

config = TPULargeConfig()
config.batch_size = 8
config.max_steps = 10000
config.requantize_every = 200
config.checkpoint_every = 500

os.makedirs("logs/jax", exist_ok=True)
t0 = time.time()
r = train(config, use_nativebit={cfg['nativebit']}, experiment_name="{cfg['name']}",
          log_dir="logs/jax", data_dir="data")
elapsed = round((time.time() - t0) / 60, 1)
result = {{"ppl": r["test_ppl"], "min": elapsed}}
with open("logs/jax/{cfg['name']}_result.json", "w") as f:
    json.dump(result, f, indent=2, default=str)
print(f"RESULT: PPL={{r['test_ppl']:.2f}} time={{elapsed}}m")
"""
    ret = subprocess.run([sys.executable, "-c", code], cwd=os.path.dirname(__file__))
    outpath = f"logs/jax/{cfg['name']}_result.json"
    if os.path.exists(outpath):
        with open(outpath) as f:
            results[cfg["name"]] = json.load(f)

# Summary
if "350m_clean_float" in results and "350m_clean_nb3" in results:
    f_ppl = results["350m_clean_float"]["ppl"]
    n_ppl = results["350m_clean_nb3"]["ppl"]
    gap = (n_ppl / f_ppl - 1) * 100
    print(f"\n{'='*60}")
    print(f"  Float: {f_ppl:.2f}  NB: {n_ppl:.2f}  Gap: {gap:+.2f}%")
    print(f"{'='*60}")

with open("logs/jax/350m_clean_results.json", "w") as f:
    json.dump(results, f, indent=2, default=str)
