import os
import sys
import csv
import json
import numpy as np
import torch
import trimesh

sys.path.insert(0, "src")

from dataset import ModelNet40Dataset
from preprocessing import MODELNET40_CLASSES, load_and_normalize_mesh
from encoder import FPS


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
DATA_DIR = "/mnt/joshi/sementic_channel_project/data/raw_data/ply_format"

OUT_DIR = (
    "/mnt/amrutha/MeshGeneration/redimensioned/runs/"
    "redimensioned_256p_8grid/occupancy_200"
)

os.makedirs(OUT_DIR, exist_ok=True)

NUM_SURFACE = 2048
NUM_PATCHES = 256
GRID_RES = 8
NUM_VOXELS = GRID_RES ** 3       # 512
PER_CLASS = 5                    # 5 × 40 = 200

np.random.seed(42)
torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


# ---------------------------------------------------------------------
# Recreate the exact validation split used by training
# We only use ds.files here; __getitem__ is NOT called, so no query
# points / occupancy labels / cache generation are needed.
# ---------------------------------------------------------------------
ds = ModelNet40Dataset(
    data_dir=DATA_DIR,
    cache_dir=OUT_DIR,
    split="val",
    num_surface=NUM_SURFACE,
    num_query=2048,
    use_augmentation=False,
    use_contrastive=False,
    seed=42,
)

print("Validation meshes:", len(ds.files))


# ---------------------------------------------------------------------
# Group validation files by class
# ---------------------------------------------------------------------
by_class = {c: [] for c in MODELNET40_CLASSES}

for path in ds.files:
    fname = os.path.basename(path)

    # Same class-name extraction used by preprocessing.py
    cls = "_".join(fname.split("_")[:-1])

    if cls in by_class:
        by_class[cls].append(path)


rng = np.random.RandomState(42)

selected = []

for cls in MODELNET40_CLASSES:
    files = list(by_class[cls])

    if len(files) < PER_CLASS:
        print("WARNING:", cls, "has only", len(files), "validation files")
        chosen = files
    else:
        idx = rng.choice(len(files), PER_CLASS, replace=False)
        chosen = [files[i] for i in idx]

    selected.extend((cls, f) for f in chosen)

print("Selected meshes:", len(selected))
print("Classes represented:", len(set(c for c, _ in selected)))


# ---------------------------------------------------------------------
# Occupancy calculation
# Uses the same FPS and voxel-coordinate formula as encoder.py.
# ---------------------------------------------------------------------
rows = []

for i, (cls, path) in enumerate(selected, start=1):

    mesh = load_and_normalize_mesh(path)

    if mesh is None:
        print("SKIP: failed to load", path)
        continue

    # Same surface sampling principle used by preprocessing.py.
    points, _ = trimesh.sample.sample_surface(mesh, NUM_SURFACE)
    points = points.astype(np.float32)

    xyz = torch.from_numpy(points).unsqueeze(0).to(device)  # [1,2048,3]

    with torch.no_grad():

        anchor_idx = FPS.forward(
            xyz,
            min(NUM_PATCHES, xyz.shape[1])
        )  # [1,256]

        anchor_pts = xyz[
            torch.arange(1, device=device).unsqueeze(1),
            anchor_idx
        ]  # [1,256,3]

        # EXACT same mapping as PointGridEncoder
        vox_coords = (
            ((anchor_pts + 0.5) / 1.0 * (GRID_RES - 1))
            .long()
            .clamp(0, GRID_RES - 1)
        )

        vox_flat = (
            vox_coords[..., 0] * GRID_RES * GRID_RES
            + vox_coords[..., 1] * GRID_RES
            + vox_coords[..., 2]
        ).reshape(-1)

        voxel_counts = torch.bincount(
            vox_flat,
            minlength=NUM_VOXELS
        )

        occupied = int((voxel_counts > 0).sum().item())
        empty = NUM_VOXELS - occupied

        collision_voxels = int((voxel_counts > 1).sum().item())
        collision_excess = NUM_PATCHES - occupied

        occupancy_ratio = occupied / NUM_VOXELS
        empty_ratio = empty / NUM_VOXELS
        collision_rate = collision_excess / NUM_PATCHES

        max_patches = int(voxel_counts.max().item())
        mean_patches_occupied = NUM_PATCHES / max(occupied, 1)

        rows.append({
            "class": cls,
            "file": path,
            "occupied_voxels": occupied,
            "empty_voxels": empty,
            "occupancy_ratio": occupancy_ratio,
            "empty_ratio": empty_ratio,
            "collision_voxels": collision_voxels,
            "collision_excess": collision_excess,
            "collision_rate": collision_rate,
            "max_patches_per_voxel": max_patches,
            "mean_patches_per_occupied_voxel": mean_patches_occupied,
        })

    if i % 20 == 0 or i == len(selected):
        print("Processed", i, "/", len(selected))


# ---------------------------------------------------------------------
# Save per-mesh results
# ---------------------------------------------------------------------
csv_path = os.path.join(OUT_DIR, "occupancy_per_mesh.csv")

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)


# ---------------------------------------------------------------------
# Dataset-level summary
# ---------------------------------------------------------------------
metrics = [
    "occupied_voxels",
    "empty_voxels",
    "occupancy_ratio",
    "empty_ratio",
    "collision_voxels",
    "collision_excess",
    "collision_rate",
    "max_patches_per_voxel",
    "mean_patches_per_occupied_voxel",
]

summary = {
    "n_meshes": len(rows),
    "num_surface_points": NUM_SURFACE,
    "num_patches": NUM_PATCHES,
    "grid_res": GRID_RES,
    "num_voxels": NUM_VOXELS,
}

for metric in metrics:
    values = np.array([r[metric] for r in rows], dtype=float)

    summary[metric] = {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "min": float(values.min()),
        "p25": float(np.percentile(values, 25)),
        "median": float(np.median(values)),
        "p75": float(np.percentile(values, 75)),
        "max": float(values.max()),
    }


# ---------------------------------------------------------------------
# Per-class occupancy mean
# ---------------------------------------------------------------------
per_class = {}

for cls in MODELNET40_CLASSES:
    cls_rows = [r for r in rows if r["class"] == cls]

    if not cls_rows:
        continue

    occ = np.array(
        [r["occupancy_ratio"] for r in cls_rows],
        dtype=float
    )

    col = np.array(
        [r["collision_rate"] for r in cls_rows],
        dtype=float
    )

    per_class[cls] = {
        "n": len(cls_rows),
        "mean_occupancy_ratio": float(occ.mean()),
        "mean_empty_ratio": float(1.0 - occ.mean()),
        "mean_collision_rate": float(col.mean()),
    }

summary["per_class"] = per_class

json_path = os.path.join(OUT_DIR, "occupancy_summary.json")

with open(json_path, "w") as f:
    json.dump(summary, f, indent=2)


# ---------------------------------------------------------------------
# Print the important result
# ---------------------------------------------------------------------
print("\n" + "=" * 65)
print("REDIMENSIONED VOXEL OCCUPANCY" , len(rows), "VALIDATION MESHES")
print("=" * 65)

for metric in [
    "occupied_voxels",
    "occupancy_ratio",
    "empty_ratio",
    "collision_voxels",
    "collision_excess",
    "collision_rate",
    "max_patches_per_voxel",
    "mean_patches_per_occupied_voxel",
]:
    s = summary[metric]

    print(
        metric,
        "\n  mean   =", round(s["mean"], 4),
        "\n  median =", round(s["median"], 4),
        "\n  min    =", round(s["min"], 4),
        "\n  max    =", round(s["max"], 4)
    )

print("\nSaved:")
print(csv_path)
print(json_path)
