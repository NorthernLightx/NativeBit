"""Text generation utilities for NativeBit models."""

import torch
import torch.nn.functional as F
import tiktoken


def generate(
    model: torch.nn.Module,
    prompt: str,
    max_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 40,
    device: torch.device = None,
    stop_at_eos: bool = True,
) -> str:
    """Generate text from a prompt using greedy/sampling decoding.

    Args:
        model: NativeBitGPT or similar causal LM.
        prompt: Input text string.
        max_tokens: Maximum tokens to generate.
        temperature: Sampling temperature (0 = greedy).
        top_k: Top-k filtering (0 = disabled).
        device: Device to run on.

    Returns:
        Generated text (prompt + completion).
    """
    if device is None:
        device = next(model.parameters()).device

    enc = tiktoken.get_encoding("gpt2")
    input_ids = enc.encode(prompt, allowed_special={"<|endoftext|>"})
    context_len = model.context_len if hasattr(model, "context_len") else 256

    idx = torch.tensor([input_ids], dtype=torch.long, device=device)
    model.set_mode_inference()

    with torch.no_grad():
        for _ in range(max_tokens):
            idx_cond = idx[:, -context_len:]

            with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(idx_cond)

            logits = logits[:, -1, :].float()

            if temperature == 0:
                next_id = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k > 0:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_id = torch.multinomial(probs, num_samples=1)

            idx = torch.cat([idx, next_id], dim=1)

            if stop_at_eos and next_id.item() == 50256:
                break

    output_ids = idx[0].tolist()
    model.train()
    return enc.decode(output_ids)


def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """Load a model from a checkpoint file."""
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    from nativebit.model import build_model_from_config

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    config_dict = ckpt.get("config", {})

    has_codebook = any("codebook" in k for k in ckpt["model_state_dict"].keys())

    class Config:
        pass
    config = Config()
    for k, v in config_dict.items():
        setattr(config, k, v)

    # Infer architecture from checkpoint tensors
    sd = ckpt["model_state_dict"]
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}

    # Infer block_size and n_codebook from codebook shapes
    if has_codebook:
        for k, v in sd.items():
            if k.endswith(".codebook") and v.dim() == 2:
                # codebook shape is (num_blocks, n_entries)
                # Find matching weight to compute block_size
                weight_key = k.replace(".codebook", ".weight")
                if weight_key in sd:
                    total_weights = sd[weight_key].numel()
                    num_blocks = v.shape[0]
                    inferred_bs = total_weights // num_blocks
                    if not hasattr(config, "block_size") or config_dict.get("block_size") is None:
                        config.block_size = inferred_bs
                    if not hasattr(config, "n_codebook") or config_dict.get("n_codebook") is None:
                        config.n_codebook = v.shape[1]
                break

    # Defaults for missing fields
    for attr, default in [
        ("vocab_size", 50257), ("n_layers", 6), ("n_embd", 256),
        ("n_head", 4), ("ffn_hidden", 1024), ("context_len", 512),
        ("block_size", 32), ("n_codebook", 8),
    ]:
        if not hasattr(config, attr):
            setattr(config, attr, default)

    # Also infer n_layers, n_embd, n_head, ffn_hidden from weights if not in config
    layer_keys = [k for k in sd if k.startswith("blocks.") and k.endswith(".attn.qkv.weight")]
    if layer_keys and "n_layers" not in config_dict:
        config.n_layers = len(layer_keys)
    for k, v in sd.items():
        if k == "blocks.0.attn.qkv.weight" and "n_embd" not in config_dict:
            config.n_embd = v.shape[1]
            config.ffn_hidden = config.n_embd * 4  # default ratio
        if k == "blocks.0.attn.qkv.weight" and "n_head" not in config_dict:
            # qkv output is 3 * n_embd, can't infer n_head directly; use default
            pass
        if k == "blocks.0.ffn.w_gate.weight" and "ffn_hidden" not in config_dict:
            config.ffn_hidden = v.shape[0]

    model = build_model_from_config(config, use_nativebit=has_codebook)
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.set_mode_inference()

    return model, config_dict
