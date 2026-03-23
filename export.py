"""CLI for exporting NativeBit training checkpoints to packed format.

Usage:
    python export.py --checkpoint logs/ts_nativebit_3bit_final.pt --output models/ts_3bit.nbpack
    python export.py --checkpoint logs/ts_nativebit_3bit_final.pt --output models/ts_3bit.nbpack --verify
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(description="Export NativeBit checkpoint to packed format")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to training checkpoint (.pt)")
    parser.add_argument("--output", type=str, required=True, help="Path for packed output (.nbpack)")
    parser.add_argument("--verify", action="store_true", help="Verify packed model matches original")
    parser.add_argument("--device", type=str, default="cpu", help="Device for computation (default: cpu)")
    args = parser.parse_args()

    from nativebit.pack import export_packed, verify_packed

    print(f"Exporting: {args.checkpoint}")
    print(f"Output:    {args.output}")
    print()

    stats = export_packed(args.checkpoint, args.output, args.device)

    # Print results
    orig_mb = stats["original_size_bytes"] / 1024 / 1024
    packed_mb = stats["file_size_bytes"] / 1024 / 1024

    print(f"Original checkpoint:  {orig_mb:.1f} MB")
    print(f"Packed file:          {packed_mb:.1f} MB")
    print(f"Compression ratio:    {stats['compression_ratio']:.1f}x")
    print(f"Bit width:            {stats['bits']}-bit")
    print(f"Quantized layers:     {stats['n_quantized_layers']}")
    print()
    print(f"  Packed indices:     {stats['packed_indices_bytes'] / 1024:.1f} KB")
    print(f"  Codebook tables:    {stats['codebook_bytes'] / 1024:.1f} KB")
    print(f"  Float params:       {stats['float_param_bytes'] / 1024:.1f} KB")

    if args.verify:
        print()
        print("Verifying packed model...")
        result = verify_packed(args.checkpoint, args.output, args.device)
        status = "PASS" if result["passed"] else "FAIL"
        print(f"  Max logit diff:   {result['max_diff']:.6f}")
        print(f"  Mean logit diff:  {result['mean_diff']:.6f}")
        print(f"  Status:           {status}")
        if not result["passed"]:
            sys.exit(1)

    print()
    print("Done.")


if __name__ == "__main__":
    main()
