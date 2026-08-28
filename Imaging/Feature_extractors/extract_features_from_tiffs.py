#!/usr/bin/env python3
"""Stage 2: extract per-frame and temporal features from saved TIFFs."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

try:
    from .segment_single_image import _parse_channel_list, extract_features_from_labels_to_csv
    from .temporal_features import build_temporal_tables
except ImportError:
    from segment_single_image import _parse_channel_list, extract_features_from_labels_to_csv
    from temporal_features import build_temporal_tables


def _channel_columns(manifest):
    pairs = []
    for col in manifest.columns:
        if col.startswith("channel_") and col.endswith("_file"):
            idx = int(col.split("_")[1])
            pairs.append((idx, col))
    return [col for _, col in sorted(pairs)]


def load_frame_from_manifest_row(row, channel_file_cols):
    channels = [tifffile.imread(row[col]) for col in channel_file_cols]
    return np.stack(channels, axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, help="CSV written by extract_czi_tiffs.py")
    parser.add_argument("--metadata", required=True, help="metadata.json written by extract_czi_tiffs.py")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--segment-channel", default=None, help="Defaults to manifest seg_channel_name or metadata channel.")
    parser.add_argument("--summarize-channels", default="all")
    parser.add_argument("--no-extended-features", action="store_true")
    parser.add_argument("--brightfield-channel", default=None)
    parser.add_argument("--oblique-channel", default=None)
    parser.add_argument("--oblique-tile-size", type=int, default=34)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--neighborhood-radii", default="50,100")
    parser.add_argument("--include-raw-crops", action="store_true")
    parser.add_argument("--out-zarr", default=None, help="Optional appendable crop store.")
    parser.add_argument("--crop-channels", default="all")
    parser.add_argument("--frame-interval", type=float, default=1.0)
    parser.add_argument("--time-unit", default="frame")
    parser.add_argument("--skip-temporal", action="store_true")
    parser.add_argument("--tracking-backend", default="simple", choices=["simple", "centroid", "btrack"])
    parser.add_argument("--btrack-config", default=None)
    parser.add_argument("--btrack-max-search-radius", type=float, default=50.0)
    parser.add_argument("--btrack-num-workers", type=int, default=1)
    parser.add_argument("--btrack-assignment-max-distance-px", type=float, default=5.0)
    parser.add_argument("--btrack-unmatched-policy", default="singleton", choices=["singleton", "keep", "error"])
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frame_feature_dir = out_dir / "frame_features"
    frame_feature_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.manifest)
    with open(args.metadata, "r") as fh:
        info = json.load(fh)
    channel_file_cols = _channel_columns(manifest)
    if not channel_file_cols:
        raise ValueError("Manifest does not contain channel_*_file columns.")

    frame_tables = []
    for row in manifest.sort_values("frame_id").itertuples(index=False):
        row = row._asdict()
        frame_id = int(row["frame_id"])
        image = load_frame_from_manifest_row(row, channel_file_cols)
        labels = tifffile.imread(row["mask_file"])
        seg_channel = args.segment_channel or row.get("seg_channel_name") or info["channels"][0]
        out_csv = frame_feature_dir / f"frame_{frame_id:06d}_features.csv"
        df = extract_features_from_labels_to_csv(
            image=image,
            labels=labels,
            info=info,
            out_csv=out_csv,
            mask_file=row["mask_file"],
            segment_channel=seg_channel,
            summarize_channels=args.summarize_channels,
            extract_extended_features=not args.no_extended_features,
            brightfield_channel=args.brightfield_channel,
            oblique_channel=args.oblique_channel,
            oblique_tile_size=args.oblique_tile_size,
            crop_size=args.crop_size,
            neighborhood_radii=[float(r) for r in _parse_channel_list(args.neighborhood_radii)],
            include_raw_crops=args.include_raw_crops,
            out_zarr=args.out_zarr,
            crop_channels=args.crop_channels,
            frame_id=frame_id,
        )
        frame_tables.append(df)

    all_features = pd.concat(frame_tables, ignore_index=True, sort=False) if frame_tables else pd.DataFrame()
    all_features_path = out_dir / "all_frame_features.csv"
    all_features.to_csv(all_features_path, index=False)
    print(f"Wrote per-frame features: {frame_feature_dir}")
    print(f"Wrote combined features: {all_features_path}")

    if not args.skip_temporal:
        tracking_kwargs = {}
        if args.tracking_backend == "btrack":
            tracking_kwargs = {
                "config_file": args.btrack_config,
                "max_search_radius": args.btrack_max_search_radius,
                "num_workers": args.btrack_num_workers,
                "assignment_max_distance_px": args.btrack_assignment_max_distance_px,
                "unmatched_policy": args.btrack_unmatched_policy,
            }
        temporal_dir = out_dir / "temporal_features"
        build_temporal_tables(
            all_features,
            frame_interval=args.frame_interval,
            time_unit=args.time_unit,
            tracking_backend=args.tracking_backend,
            tracking_kwargs=tracking_kwargs,
            output_dir=temporal_dir,
        )
        print(f"Wrote temporal features: {temporal_dir}")


if __name__ == "__main__":
    main()
