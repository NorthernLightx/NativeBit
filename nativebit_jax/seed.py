"""JAX PRNG key management."""

import jax


def make_rng(seed: int = 42) -> jax.Array:
    return jax.random.PRNGKey(seed)


def split(rng: jax.Array, n: int = 2):
    return jax.random.split(rng, n)
