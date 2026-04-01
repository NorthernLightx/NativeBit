"""WikiText-2 data loading with tiktoken tokenizer."""

import os
import urllib.request
import zipfile
from pathlib import Path

import torch
import tiktoken


# Download individual splits from PyTorch examples repo
_GITHUB_BASE = "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2"
_SPLITS = {
    "train": "train.txt",
    "valid": "valid.txt",
    "test": "test.txt",
}
# Map our expected filenames to the download filenames
_TOKEN_FILES = {
    "wiki.train.tokens": "train.txt",
    "wiki.valid.tokens": "valid.txt",
    "wiki.test.tokens": "test.txt",
}


def _download_file(url: str, dest: Path, timeout: int = 60) -> None:
    """Download a URL to a local file."""
    req = urllib.request.Request(url, headers={"User-Agent": "NativeBit/0.1"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        with open(dest, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)


def download_wikitext2(data_dir: str = "data") -> Path:
    """Download WikiText-2 splits if not present. Returns the dataset directory."""
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "wikitext-2"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check if all token files exist (either original or downloaded names)
    all_present = all(
        (dataset_dir / tok_name).exists() or (dataset_dir / dl_name).exists()
        for tok_name, dl_name in _TOKEN_FILES.items()
    )

    if all_present:
        return dataset_dir

    # Download each split
    for split_name, filename in _SPLITS.items():
        dest = dataset_dir / f"wiki.{split_name}.tokens"
        if dest.exists():
            continue
        url = f"{_GITHUB_BASE}/{filename}"
        print(f"Downloading WikiText-2 {split_name} from {url}...")
        _download_file(url, dest)

    return dataset_dir


def _clean_wikitext(text: str) -> str:
    """Clean WikiText-2 preprocessing artifacts.

    WikiText-2 has several artifacts from its preprocessing pipeline:
    - <unk> for unknown words
    - @-@ for hyphens (e.g. "well @-@ known" -> "well-known")
    - @.@ for decimal points (e.g. "3 @.@ 14" -> "3.14")
    - @,@ for commas in numbers (e.g. "1 @,@ 000" -> "1,000")
    - = Section Headers = as article boundaries
    """
    text = text.replace(" @-@ ", "-")
    text = text.replace(" @.@ ", ".")
    text = text.replace(" @,@ ", ",")
    text = text.replace("<unk>", "")
    # Collapse multiple spaces from <unk> removal
    while "  " in text:
        text = text.replace("  ", " ")
    return text


def load_wikitext2_tokens(split: str = "train", data_dir: str = "data") -> torch.Tensor:
    """Load and tokenize a WikiText-2 split using tiktoken gpt2 encoding.

    Args:
        split: one of 'train', 'valid', 'test'.
        data_dir: path to data directory.

    Returns:
        1-D tensor of token ids.
    """
    dataset_dir = download_wikitext2(data_dir)

    fname = {"train": "wiki.train.tokens", "valid": "wiki.valid.tokens", "test": "wiki.test.tokens"}
    path = dataset_dir / fname[split]

    text = path.read_text(encoding="utf-8")
    text = _clean_wikitext(text)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text, allowed_special=set())

    return torch.tensor(tokens, dtype=torch.long)


class BatchIterator:
    """Fast batch iterator with precomputed sequence tensor.

    Precomputes all (x, y) pairs as a contiguous tensor at init time,
    so iteration is just tensor indexing — no Python loops per batch.
    """

    def __init__(self, tokens: torch.Tensor, context_len: int, batch_size: int,
                 shuffle: bool = False, drop_last: bool = False):
        self.context_len = context_len
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.n_sequences = (len(tokens) - 1) // context_len

        # Precompute all sequences as (n_sequences, context_len) tensors
        n = self.n_sequences
        # x[i] = tokens[i*ctx : i*ctx + ctx], y[i] = tokens[i*ctx+1 : i*ctx+ctx+1]
        all_x = tokens[: n * context_len].view(n, context_len)
        all_y = tokens[1 : n * context_len + 1].view(n, context_len)
        self.all_x = all_x
        self.all_y = all_y

    def __len__(self) -> int:
        if self.drop_last:
            return self.n_sequences // self.batch_size
        return (self.n_sequences + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        if self.shuffle:
            perm = torch.randperm(self.n_sequences)
            x = self.all_x[perm]
            y = self.all_y[perm]
        else:
            x = self.all_x
            y = self.all_y

        for start in range(0, self.n_sequences, self.batch_size):
            end = start + self.batch_size
            if self.drop_last and end > self.n_sequences:
                break
            end = min(end, self.n_sequences)
            yield x[start:end], y[start:end]


def download_wikitext103(data_dir: str = "data") -> Path:
    """Download WikiText-103 splits if not present. Returns the dataset directory."""
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "wikitext-103"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    if (dataset_dir / "wiki.train.tokens").exists():
        return dataset_dir

    print("Downloading WikiText-103 via HuggingFace datasets...")
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-v1", trust_remote_code=True)

    for split, fname in [("train", "wiki.train.tokens"),
                         ("validation", "wiki.valid.tokens"),
                         ("test", "wiki.test.tokens")]:
        dest = dataset_dir / fname
        text = "\n".join(row["text"] for row in ds[split])
        dest.write_text(text, encoding="utf-8")
        print(f"  Saved {fname} ({len(text) // 1024 // 1024} MB)")

    return dataset_dir


def load_wikitext103_tokens(split: str = "train", data_dir: str = "data") -> torch.Tensor:
    """Load and tokenize a WikiText-103 split using tiktoken gpt2 encoding."""
    dataset_dir = download_wikitext103(data_dir)

    fname = {"train": "wiki.train.tokens", "valid": "wiki.valid.tokens", "test": "wiki.test.tokens"}
    path = dataset_dir / fname[split]

    text = path.read_text(encoding="utf-8")
    text = _clean_wikitext(text)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(tokens, dtype=torch.long)


def download_tinystories(data_dir: str = "data") -> Path:
    """Download TinyStories dataset if not present. Returns the dataset directory.

    Downloads the validated subset from HuggingFace as raw text.
    """
    data_dir = Path(data_dir)
    dataset_dir = data_dir / "tinystories"
    dataset_dir.mkdir(parents=True, exist_ok=True)

    train_path = dataset_dir / "train.txt"
    valid_path = dataset_dir / "valid.txt"

    if train_path.exists() and valid_path.exists():
        return dataset_dir

    # Download from HuggingFace datasets (raw parquet -> text extraction)
    # TinyStories is small enough to download as text files
    _TS_BASE = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main"
    for fname, dest in [("TinyStoriesV2-GPT4-train.txt", train_path),
                        ("TinyStoriesV2-GPT4-valid.txt", valid_path)]:
        if dest.exists():
            continue
        url = f"{_TS_BASE}/{fname}"
        print(f"Downloading TinyStories {fname}...")
        _download_file(url, dest, timeout=300)

    return dataset_dir


def load_tinystories_tokens(split: str = "train", data_dir: str = "data",
                            max_chars: int = 50_000_000) -> torch.Tensor:
    """Load and tokenize TinyStories split using tiktoken gpt2 encoding.

    Args:
        split: "train" or "valid" (valid is used for both valid and test).
        data_dir: path to data directory.
        max_chars: max characters to load (TinyStories is large).

    Returns:
        1-D tensor of token ids.
    """
    dataset_dir = download_tinystories(data_dir)
    fname = "train.txt" if split == "train" else "valid.txt"
    path = dataset_dir / fname

    # Read limited amount (full TinyStories is ~2GB)
    with open(path, encoding="utf-8") as f:
        text = f.read(max_chars)

    enc = tiktoken.get_encoding("gpt2")
    tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
    return torch.tensor(tokens, dtype=torch.long)


def get_dataloaders(
    context_len: int,
    batch_size: int,
    data_dir: str = "data",
    dataset: str = "wikitext-2",
) -> tuple:
    """Get train, validation, and test batch iterators.

    Args:
        dataset: "wikitext-2" or "tinystories"

    Returns:
        (train_loader, valid_loader, test_loader)
    """
    if dataset == "tinystories":
        train_tokens = load_tinystories_tokens("train", data_dir)
        valid_tokens = load_tinystories_tokens("valid", data_dir)
        # Use second half of valid as test
        mid = len(valid_tokens) // 2
        test_tokens = valid_tokens[mid:]
        valid_tokens = valid_tokens[:mid]
    elif dataset == "wikitext-103":
        train_tokens = load_wikitext103_tokens("train", data_dir)
        valid_tokens = load_wikitext103_tokens("valid", data_dir)
        test_tokens = load_wikitext103_tokens("test", data_dir)
    else:
        train_tokens = load_wikitext2_tokens("train", data_dir)
        valid_tokens = load_wikitext2_tokens("valid", data_dir)
        test_tokens = load_wikitext2_tokens("test", data_dir)

    train_loader = BatchIterator(train_tokens, context_len, batch_size, shuffle=True, drop_last=True)
    valid_loader = BatchIterator(valid_tokens, context_len, batch_size, shuffle=False)
    test_loader = BatchIterator(test_tokens, context_len, batch_size, shuffle=False)

    print(f"[{dataset}] Train: {train_loader.n_sequences} seq, Valid: {valid_loader.n_sequences}, Test: {test_loader.n_sequences}")
    return train_loader, valid_loader, test_loader


def build_token_byte_table() -> torch.Tensor:
    """Precompute UTF-8 byte length for each token in tiktoken gpt2 vocab.

    Returns:
        Tensor of shape (vocab_size,) with int32 byte lengths.
    """
    enc = tiktoken.get_encoding("gpt2")
    byte_lengths = []
    for token_id in range(enc.n_vocab):
        try:
            decoded = enc.decode([token_id])
            byte_lengths.append(len(decoded.encode("utf-8")))
        except Exception:
            byte_lengths.append(0)
    return torch.tensor(byte_lengths, dtype=torch.int32)


_TOKEN_BYTE_TABLE = None

def _get_token_byte_table(device: torch.device) -> torch.Tensor:
    """Get cached token byte table, moved to device."""
    global _TOKEN_BYTE_TABLE
    if _TOKEN_BYTE_TABLE is None:
        _TOKEN_BYTE_TABLE = build_token_byte_table()
    return _TOKEN_BYTE_TABLE.to(device)


@torch.no_grad()
def compute_bpb(model, loader, device: torch.device) -> float:
    """Compute bits per byte (BPB) -- vocab-size-independent metric.

    BPB = total_nats / (ln(2) * total_bytes)

    Args:
        model: language model (must be in eval mode).
        loader: BatchIterator yielding (x, y) pairs.
        device: torch device.

    Returns:
        BPB as float (lower is better).
    """
    import torch.nn.functional as F
    import math

    from nativebit.device import amp_context, mark_step

    token_bytes = _get_token_byte_table(device)
    total_nats = torch.tensor(0.0, device=device)
    total_bytes_t = torch.tensor(0, dtype=torch.long, device=device)

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with amp_context(device):
            logits = model(x)
            loss_flat = F.cross_entropy(
                logits.view(-1, logits.size(-1)), y.view(-1), reduction='none'
            )

        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat.float() * mask).sum()
        total_bytes_t += nbytes.sum()
        mark_step()

    total_bytes_val = total_bytes_t.item()
    if total_bytes_val == 0:
        return float('inf')
    return total_nats.item() / (math.log(2) * total_bytes_val)
