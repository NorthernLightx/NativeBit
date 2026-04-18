"""Minimal FSDP test — verify sharding works on this JAX/TPU runtime."""
import jax
import jax.numpy as jnp
import optax
from jax.sharding import Mesh, PartitionSpec as P, NamedSharding

print(f"JAX {jax.__version__}, backend={jax.default_backend()}, devices={jax.device_count()}")

devices = jax.devices()
mesh = Mesh(devices, axis_names=('fsdp',))

# Test 1: basic sharded matmul
print("\n--- Test 1: sharded matmul ---")
w = jnp.ones((2560, 2560), dtype=jnp.float32)
w = jax.device_put(w, NamedSharding(mesh, P('fsdp', None)))
x = jnp.ones((32, 2048, 2560), dtype=jnp.bfloat16)
x = jax.device_put(x, NamedSharding(mesh, P('fsdp', None, None)))

@jax.jit
def matmul(w, x):
    return x @ w.T

result = matmul(w, x)
print(f"  w sharding: {w.sharding}")
print(f"  x sharding: {x.sharding}")
print(f"  result sharding: {result.sharding}")
print(f"  result shape: {result.shape}")

# Test 2: grad through sharded matmul
print("\n--- Test 2: grad through sharded matmul ---")
@jax.jit
def loss_fn(w, x):
    y = x.astype(jnp.bfloat16) @ w.astype(jnp.bfloat16).T
    return jnp.mean(y)

grad_fn = jax.jit(jax.grad(loss_fn))
g = grad_fn(w, x)
print(f"  grad sharding: {g.sharding}")
print(f"  grad shape: {g.shape}")

# Test 3: optax optimizer with sharded params
print("\n--- Test 3: optax with sharded params ---")
tx = optax.adamw(1e-3)
opt_state = tx.init(w)
print(f"  opt_state mu sharding: {opt_state[0].mu.sharding}")

# Test 4: full train step
print("\n--- Test 4: full train step ---")
@jax.jit
def train_step(params, opt_state, x):
    def loss(p):
        return jnp.mean(x.astype(jnp.bfloat16) @ p.astype(jnp.bfloat16).T)
    l, g = jax.value_and_grad(loss)(params)
    updates, new_opt = tx.update(g, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt, l

w2, opt2, loss_val = train_step(w, opt_state, x)
print(f"  loss: {float(loss_val):.4f}")
print(f"  updated w sharding: {w2.sharding}")
print(f"  updated opt sharding: {opt2[0].mu.sharding}")

# Test 5: memory usage
print("\n--- Test 5: memory check ---")
for i, d in enumerate(devices):
    stats = d.memory_stats()
    used = stats['bytes_in_use'] / 1e9
    limit = stats['bytes_limit'] / 1e9
    print(f"  Device {i}: {used:.2f}GB / {limit:.2f}GB")

print("\nAll FSDP tests passed!")
