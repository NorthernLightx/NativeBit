"""Quick benchmark to find training bottleneck."""
import torch
import time
from nativebit.model import build_model_from_config
from configs.small import SmallConfig

config = SmallConfig()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Benchmark float baseline
model = build_model_from_config(config, use_nativebit=False).to(device)
x = torch.randint(0, config.vocab_size, (config.batch_size, config.context_len), device=device)
y = torch.randint(0, config.vocab_size, (config.batch_size, config.context_len), device=device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Warmup
for _ in range(3):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.time()
n = 20
for _ in range(n):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f"Float model: {n/elapsed:.1f} steps/sec ({elapsed/n*1000:.0f} ms/step)")

# Benchmark NativeBit
del model
if device.type == "cuda":
    torch.cuda.empty_cache()

model = build_model_from_config(config, use_nativebit=True).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

for _ in range(3):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()

start = time.time()
for _ in range(n):
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
if device.type == "cuda":
    torch.cuda.synchronize()
elapsed = time.time() - start
print(f"NativeBit model: {n/elapsed:.1f} steps/sec ({elapsed/n*1000:.0f} ms/step)")
if device.type == "cuda":
    print(f"GPU memory: {torch.cuda.max_memory_allocated()/1e6:.0f} MB")
