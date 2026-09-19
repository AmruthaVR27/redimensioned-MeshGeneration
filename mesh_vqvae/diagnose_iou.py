import sys
import torch
import numpy as np
from torch.utils.data import DataLoader

sys.path.insert(0, "src")

from config import SmallModelConfig
from model import MaskedVQVAE3D
from dataset import ModelNet40Dataset


CKPT = (
    "/mnt/amrutha/MeshGeneration/redimensioned/runs/"
    "redimensioned_256p_8grid_full/checkpoints/last.ckpt"
)

DATA_DIR = "/mnt/joshi/sementic_channel_project/data/raw_data/ply_format"

CACHE_DIR = (
    "/mnt/amrutha/MeshGeneration/redimensioned/runs/"
    "redimensioned_256p_8grid_full/cache"
)

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7]


# ------------------------------------------------------------
# Load redimensioned model
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cfg = SmallModelConfig()

model = MaskedVQVAE3D(cfg)

ckpt = torch.load(CKPT, map_location="cpu")
model.load_state_dict(ckpt["state_dict"])

model.to(device)
model.eval()

print("Checkpoint epoch:", ckpt.get("epoch"))
print("Device:", device)
print("Grid:", cfg.encoder.grid_res, "^3")
print("Patches:", cfg.encoder.num_patches)


# ------------------------------------------------------------
# Full validation dataset
# ------------------------------------------------------------
val_ds = ModelNet40Dataset(
    DATA_DIR,
    CACHE_DIR,
    split="val",
    num_surface=2048,
    num_query=2048,
    use_augmentation=False,
    use_contrastive=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
    num_workers=4,
    pin_memory=True,
)

print("Validation meshes:", len(val_ds))


# ------------------------------------------------------------
# Statistics
# ------------------------------------------------------------
total_targets = 0.0
total_elements = 0

prob_sum = 0.0
prob_count = 0
prob_min = float("inf")
prob_max = float("-inf")

pred_positive = {t: 0.0 for t in THRESHOLDS}
iou_values = {t: [] for t in THRESHOLDS}


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
with torch.no_grad():

    for batch_idx, batch in enumerate(val_loader):

        points = batch["points"].to(device)
        normals = batch["normals"].to(device)
        curvature = batch["curvature"].to(device)
        query_pts = batch["query_pts"].to(device)
        labels = batch["label"].to(device)
        targets = batch["occupancy"].to(device).float()

        out = model(
            points,
            normals,
            curvature,
            query_pts,
            labels,
        )

        probs = out["logits"].sigmoid()

        # Ground-truth occupancy
        total_targets += targets.sum().item()
        total_elements += targets.numel()

        # Probability distribution
        prob_sum += probs.sum().item()
        prob_count += probs.numel()
        prob_min = min(prob_min, probs.min().item())
        prob_max = max(prob_max, probs.max().item())

        # Threshold-specific statistics
        for t in THRESHOLDS:

            preds = (probs > t).float()

            pred_positive[t] += preds.sum().item()

            intersection = (preds * targets).sum(dim=1)
            union = ((preds + targets) > 0).float().sum(dim=1)

            batch_ious = intersection / (union + 1e-8)

            iou_values[t].extend(
                batch_ious.detach().cpu().numpy().tolist()
            )

        if (batch_idx + 1) % 50 == 0:
            print(
                "Processed",
                min((batch_idx + 1) * 4, len(val_ds)),
                "/",
                len(val_ds),
            )


# ------------------------------------------------------------
# Results
# ------------------------------------------------------------
true_positive_fraction = total_targets / total_elements
mean_probability = prob_sum / prob_count

print("\n" + "=" * 65)
print("RECONSTRUCTION / IoU DIAGNOSTIC")
print("=" * 65)

print(
    "\nTrue occupied-label fraction:",
    round(true_positive_fraction, 6),
    "(" + str(round(true_positive_fraction * 100, 2)) + "%)"
)

print("\nPredicted probability:")
print("  mean =", round(mean_probability, 6))
print("  min  =", round(prob_min, 6))
print("  max  =", round(prob_max, 6))

print("\nThreshold results:")

for t in THRESHOLDS:

    predicted_fraction = pred_positive[t] / total_elements
    mean_iou = float(np.mean(iou_values[t]))

    print("\nThreshold =", t)
    print(
        "  predicted occupied =",
        round(predicted_fraction, 6),
        "(" + str(round(predicted_fraction * 100, 2)) + "%)"
    )
    print(
        "  mean IoU           =",
        round(mean_iou, 9)
    )