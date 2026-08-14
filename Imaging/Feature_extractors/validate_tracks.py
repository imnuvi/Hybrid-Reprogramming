#!/usr/bin/env python3
"""Validate and visualize segmentation/track outputs."""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from skimage.segmentation import find_boundaries


def _norm(img):
    img = np.asarray(img, dtype=float)
    finite = np.isfinite(img)
    if not finite.any():
        return np.zeros(img.shape, dtype=float)
    lo, hi = np.nanpercentile(img[finite], [1, 99.8])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def _channel_file_col(manifest, channel_name=None, channel_index=0):
    if channel_name is not None:
        for i in range(100):
            name_col = f"channel_{i}_name"
            file_col = f"channel_{i}_file"
            if name_col in manifest and file_col in manifest:
                if str(manifest[name_col].iloc[0]).lower() == str(channel_name).lower():
                    return file_col
    file_col = f"channel_{int(channel_index)}_file"
    if file_col not in manifest:
        raise ValueError(f"Could not find {file_col} in manifest.")
    return file_col


def validate_masks(observations):
    rows = []
    required = {"frame_id", "label", "mask_file"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"Observation table is missing required columns: {missing}")
    for frame_id, g in observations.groupby("frame_id", sort=True):
        paths = g["mask_file"].dropna().unique()
        row = {"frame_id": int(frame_id), "n_rows": len(g), "n_mask_files": len(paths)}
        if len(paths) != 1:
            row.update({"mask_exists": False, "n_mask_labels": np.nan, "labels_missing_from_mask": len(g)})
            rows.append(row)
            continue
        path = Path(paths[0])
        row["mask_file"] = str(path)
        row["mask_exists"] = path.exists()
        if path.exists():
            mask = tifffile.imread(path)
            mask_labels = set(np.unique(mask).astype(int))
            mask_labels.discard(0)
            csv_labels = set(g["label"].astype(int))
            row["n_mask_labels"] = len(mask_labels)
            row["labels_missing_from_mask"] = len(csv_labels.difference(mask_labels))
            row["extra_mask_labels_not_in_csv"] = len(mask_labels.difference(csv_labels))
        rows.append(row)
    return pd.DataFrame(rows)


def track_qc(observations, max_jump_um=None):
    rows = []
    duplicate_track_frames = int(observations.duplicated(["track_id", "frame_id"]).sum()) if {"track_id", "frame_id"}.issubset(observations) else np.nan
    for tid, g in observations.groupby("track_id", sort=True):
        g = g.sort_values("elapsed_time")
        jumps = g["step_distance_um"].dropna() if "step_distance_um" in g else pd.Series(dtype=float)
        row = {
            "track_id": int(tid),
            "n_observations": len(g),
            "start_frame": int(g["frame_id"].min()),
            "end_frame": int(g["frame_id"].max()),
            "n_gaps": int((g["gap_frames"] > 0).sum()) if "gap_frames" in g else np.nan,
            "max_step_distance_um": float(jumps.max()) if len(jumps) else np.nan,
            "median_step_distance_um": float(jumps.median()) if len(jumps) else np.nan,
            "mean_speed": float(g["speed_um_per_time"].mean()) if "speed_um_per_time" in g else np.nan,
        }
        if max_jump_um is not None and len(jumps):
            row["has_large_jump"] = bool((jumps > max_jump_um).any())
        rows.append(row)
    qc = pd.DataFrame(rows)
    qc.attrs["duplicate_track_frames"] = duplicate_track_frames
    return qc


def write_plots(observations, track_table, frame_qc, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    counts = observations.groupby("frame_id").size()
    plt.figure(figsize=(8, 4))
    counts.plot(marker="o", linewidth=1)
    plt.xlabel("Frame")
    plt.ylabel("Cell count")
    plt.tight_layout()
    plt.savefig(out_dir / "cell_count_over_time.png", dpi=200)
    plt.close()

    if not track_table.empty:
        plt.figure(figsize=(7, 4))
        track_table["n_observations"].hist(bins=40)
        plt.xlabel("Track observations")
        plt.ylabel("Track count")
        plt.tight_layout()
        plt.savefig(out_dir / "track_length_histogram.png", dpi=200)
        plt.close()

    if "speed_um_per_time" in observations:
        plt.figure(figsize=(7, 4))
        observations["speed_um_per_time"].dropna().hist(bins=60)
        plt.xlabel("Speed")
        plt.ylabel("Observation count")
        plt.tight_layout()
        plt.savefig(out_dir / "speed_histogram.png", dpi=200)
        plt.close()

    plt.figure(figsize=(7, 7))
    for _, g in observations.groupby("track_id"):
        g = g.sort_values("frame_id")
        if len(g) >= 2:
            plt.plot(g["centroid_x_px"], g["centroid_y_px"], linewidth=0.8, alpha=0.5)
    plt.gca().invert_yaxis()
    plt.xlabel("x px")
    plt.ylabel("y px")
    plt.tight_layout()
    plt.savefig(out_dir / "track_xy_overview.png", dpi=250)
    plt.close()


def write_overlay_frames(observations, manifest, out_dir, channel_name=None, channel_index=0, max_frames=12):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    file_col = _channel_file_col(manifest, channel_name, channel_index)
    frames = sorted(observations["frame_id"].unique())
    if len(frames) > max_frames:
        idx = np.linspace(0, len(frames) - 1, max_frames).round().astype(int)
        frames = [frames[i] for i in idx]

    for frame_id in frames:
        obs = observations[observations["frame_id"] == frame_id].copy()
        man = manifest[manifest["frame_id"] == frame_id]
        if man.empty:
            continue
        raw = tifffile.imread(man[file_col].iloc[0])
        mask = tifffile.imread(obs["mask_file"].dropna().iloc[0])
        base = np.dstack([_norm(raw)] * 3) * 0.7
        boundaries = find_boundaries(mask, mode="outer")
        base[boundaries] = (1, 1, 0)

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(base)
        for _, row in obs.iterrows():
            ax.text(
                row["centroid_x_px"],
                row["centroid_y_px"],
                str(int(row["track_id"])),
                color="white",
                fontsize=5,
                ha="center",
                va="center",
            )
        ax.set_title(f"Frame {frame_id}: boundaries and track IDs")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / f"frame_{int(frame_id):06d}_tracks_overlay.png", dpi=220)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--observations", required=True, help="cell_frame_observations.csv")
    parser.add_argument("--manifest", required=True, help="tiff_manifest.csv")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--channel-name", default=None)
    parser.add_argument("--channel-index", type=int, default=0)
    parser.add_argument("--max-jump-um", type=float, default=None)
    parser.add_argument("--max-overlay-frames", type=int, default=12)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    observations = pd.read_csv(args.observations)
    manifest = pd.read_csv(args.manifest)

    frame_qc = validate_masks(observations)
    tqc = track_qc(observations, max_jump_um=args.max_jump_um)
    frame_qc.to_csv(out_dir / "frame_mask_qc.csv", index=False)
    tqc.to_csv(out_dir / "track_qc.csv", index=False)

    summary = {
        "n_observations": int(len(observations)),
        "n_frames": int(observations["frame_id"].nunique()),
        "n_tracks": int(observations["track_id"].nunique()),
        "duplicate_track_frame_rows": int(tqc.attrs.get("duplicate_track_frames", 0)),
        "frames_with_missing_masks": int((~frame_qc["mask_exists"].astype(bool)).sum()) if "mask_exists" in frame_qc else None,
        "frames_with_label_mismatches": int(((frame_qc.get("labels_missing_from_mask", 0) > 0) | (frame_qc.get("extra_mask_labels_not_in_csv", 0) > 0)).sum()),
        "median_track_observations": float(tqc["n_observations"].median()) if not tqc.empty else np.nan,
        "short_tracks_len1": int((tqc["n_observations"] == 1).sum()) if not tqc.empty else 0,
    }
    if args.max_jump_um is not None and "has_large_jump" in tqc:
        summary["tracks_with_large_jumps"] = int(tqc["has_large_jump"].sum())
    (out_dir / "validation_summary.json").write_text(json.dumps(summary, indent=2))

    write_plots(observations, tqc, frame_qc, out_dir / "plots")
    write_overlay_frames(
        observations,
        manifest,
        out_dir / "track_overlays",
        channel_name=args.channel_name,
        channel_index=args.channel_index,
        max_frames=args.max_overlay_frames,
    )
    print(f"Wrote validation outputs: {out_dir}")


if __name__ == "__main__":
    main()
