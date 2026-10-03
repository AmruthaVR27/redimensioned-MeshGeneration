import argparse
import torch


# Buffers saved in state_dict but NOT learnable parameters
BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)


def get_state_dict(checkpoint):
    """
    Handles common checkpoint formats:
      - PyTorch Lightning: checkpoint['state_dict']
      - {'model_state_dict': ...}
      - {'model': ...}
      - raw state_dict
    """
    if isinstance(checkpoint, dict):

        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            return checkpoint["state_dict"]

        if "model_state_dict" in checkpoint and isinstance(checkpoint["model_state_dict"], dict):
            return checkpoint["model_state_dict"]

        if "model" in checkpoint and isinstance(checkpoint["model"], dict):
            return checkpoint["model"]

        # raw state_dict
        if checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise RuntimeError(
        "Could not automatically find the model state_dict in this checkpoint."
    )


def count_checkpoint(name, path):
    print(f"\n{'=' * 70}")
    print(f"{name}")
    print(f"Checkpoint: {path}")
    print("=" * 70)

    ckpt = torch.load(path, map_location="cpu")
    state = get_state_dict(ckpt)

    parameter_tensors = []

    for key, tensor in state.items():

        if not torch.is_tensor(tensor):
            continue

        # Remove persistent buffers such as BatchNorm statistics
        if key.endswith(BUFFER_SUFFIXES):
            continue

        parameter_tensors.append((key, tensor))

    # -------------------------------------------------------
    # Deduplicate tensors that share storage.
    # Important for things like tied token embedding/lm_head
    # in GPT-style DoT models.
    # -------------------------------------------------------
    seen = set()
    unique_total = 0

    for key, tensor in parameter_tensors:

        try:
            identity = (
                tensor.untyped_storage().data_ptr(),
                tensor.storage_offset(),
                tensor.numel(),
            )
        except Exception:
            identity = (key,)

        if identity not in seen:
            seen.add(identity)
            unique_total += tensor.numel()

    raw_total = sum(t.numel() for _, t in parameter_tensors)

    print(f"Parameter tensors : {len(parameter_tensors):,}")
    print(f"Unique parameters : {unique_total:,}")
    print(f"Parameters (M)    : {unique_total / 1e6:.3f} M")

    if raw_total != unique_total:
        print(
            f"Raw state_dict     : {raw_total:,} "
            f"({raw_total / 1e6:.3f} M)"
        )
        print("Note: shared/tied tensors were counted only once.")

    return unique_total


def main():
    parser = argparse.ArgumentParser(
        description="Count MeshGeneration VQ-VAE, DoT and SEDD parameters."
    )

    parser.add_argument("--vqvae", required=True,
                        help="Path to VQ-VAE checkpoint")
    parser.add_argument("--dot", required=True,
                        help="Path to DoT checkpoint")
    parser.add_argument("--sedd", required=True,
                        help="Path to SEDD checkpoint")

    args = parser.parse_args()

    vq = count_checkpoint("VQ-VAE", args.vqvae)
    dot = count_checkpoint("DoT", args.dot)
    sedd = count_checkpoint("SEDD", args.sedd)

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    print(f"{'VQ-VAE':15s}: {vq:12,d}   ({vq/1e6:8.3f} M)")
    print(f"{'DoT':15s}: {dot:12,d}   ({dot/1e6:8.3f} M)")
    print(f"{'SEDD':15s}: {sedd:12,d}   ({sedd/1e6:8.3f} M)")

    print()
    print(
        f"{'VQ-VAE + DoT':15s}: "
        f"{vq + dot:12,d}   ({(vq + dot)/1e6:8.3f} M)"
    )

    print(
        f"{'VQ-VAE + SEDD':15s}: "
        f"{vq + sedd:12,d}   ({(vq + sedd)/1e6:8.3f} M)"
    )


if __name__ == "__main__":
    main()
