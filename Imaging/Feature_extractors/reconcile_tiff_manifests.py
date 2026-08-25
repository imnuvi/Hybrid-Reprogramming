#!/usr/bin/env python3
"""Combine TIFF manifests from split CZI acquisitions into one time series.

The stitched channel TIFFs and label TIFFs stay where they are.  This script
rewrites only the manifest frame ids so downstream feature extraction sees one
continuous movie.
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def _file_columns(df):
    cols = [c for c in df.columns if c == "mask_file"]
    cols.extend(c for c in df.columns if c.startswith("channel_") and c.endswith("_file"))
    return cols


def _absolutize_paths(df, manifest_path):
    base = Path(manifest_path).resolve().parent
    out = df.copy()
    for col in _file_columns(out):
        out[col] = out[col].map(lambda p: str((base / p).resolve()) if pd.notna(p) and not Path(str(p)).is_absolute() else p)
    return out


def _channel_signature(df):
    names = []
    for col in sorted([c for c in df.columns if c.startswith("channel_") and c.endswith("_name")], key=lambda x: int(x.split("_")[1])):
        names.append(str(df[col].dropna().iloc[0]) if not df[col].dropna().empty else "")
    return names


def combine_manifests(manifest_paths, run_names=None, offsets=None, start_frame=0, absolutize=True):
    if run_names is None:
        run_names = [Path(p).resolve().parent.name for p in manifest_paths]
    if len(run_names) != len(manifest_paths):
        raise ValueError("run_names must match the number of manifests.")
    if offsets is not None and len(offsets) != len(manifest_paths):
        raise ValueError("offsets must match the number of manifests.")

    combined = []
    next_frame = int(start_frame)
    expected_channels = None

    for i, manifest_path in enumerate(manifest_paths):
        manifest_path = Path(manifest_path)
        df = pd.read_csv(manifest_path).sort_values("frame_id").reset_index(drop=True)
        if absolutize:
            df = _absolutize_paths(df, manifest_path)

        channels = _channel_signature(df)
        if expected_channels is None:
            expected_channels = channels
        elif channels != expected_channels:
            raise ValueError(f"Channel names differ in {manifest_path}: {channels} != {expected_channels}")

        source_frame = df["frame_id"].astype(int)
        offset = int(offsets[i]) if offsets is not None else next_frame - int(source_frame.min())
        df["source_manifest"] = str(manifest_path.resolve())
        df["source_run"] = run_names[i]
        df["source_frame_id"] = source_frame
        df["frame_id"] = source_frame + offset
        df["frame_offset_applied"] = offset
        combined.append(df)
        next_frame = int(df["frame_id"].max()) + 1

    out = pd.concat(combined, ignore_index=True, sort=False).sort_values("frame_id").reset_index(drop=True)
    if out["frame_id"].duplicated().any():
        dupes = out.loc[out["frame_id"].duplicated(), "frame_id"].tolist()
        raise ValueError(f"Combined manifest has duplicate frame_id values: {dupes[:10]}")
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifests", nargs="+", required=True, help="tiff_manifest.csv files in chronological order.")
    parser.add_argument("--metadata-files", nargs="*", default=None, help="Optional metadata.json files in the same order.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--run-names", default=None, help="Comma-separated names, e.g. czi0_24,czi24_72.")
    parser.add_argument("--offsets", default=None, help="Comma-separated frame offsets. Default: append each manifest after the previous one.")
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--keep-relative-paths", action="store_true")
    args = parser.parse_args()

    run_names = None if args.run_names is None else [x.strip() for x in args.run_names.split(",") if x.strip()]
    offsets = None if args.offsets is None else [int(x) for x in args.offsets.split(",")]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    combined = combine_manifests(
        args.manifests,
        run_names=run_names,
        offsets=offsets,
        start_frame=args.start_frame,
        absolutize=not args.keep_relative_paths,
    )
    manifest_out = out_dir / "combined_tiff_manifest.csv"
    combined.to_csv(manifest_out, index=False)

    metadata = {}
    if args.metadata_files:
        if len(args.metadata_files) != len(args.manifests):
            raise ValueError("--metadata-files must match --manifests length.")
        for i, path in enumerate(args.metadata_files):
            with open(path, "r") as fh:
                metadata[f"source_{i}"] = json.load(fh)
        metadata["combined_metadata_note"] = "Use source_0 as the feature-extraction metadata if channels/pixel sizes are identical."
        metadata_out = out_dir / "combined_metadata_sources.json"
        metadata_out.write_text(json.dumps(metadata, indent=2))
    print(f"Wrote combined manifest: {manifest_out}")


if __name__ == "__main__":
    main()
