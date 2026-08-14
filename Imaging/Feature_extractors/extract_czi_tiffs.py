#!/usr/bin/env python3
"""Stage 1: extract stitched channel TIFFs and segmentation label TIFFs from a CZI.

This script writes a manifest CSV that is consumed by
``extract_features_from_tiffs.py``.  The manifest keeps the raw stitched channel
TIFFs, label-mask TIFFs, frame ids, scene, and channel names tied together.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from aicspylibczi import CziFile

try:
    from .segment_single_image import segment_image_to_labels, write_label_tiff
except ImportError:
    from segment_single_image import segment_image_to_labels, write_label_tiff


def _normalize_tile(tile, q_low=1, q_high=99.8):
    tile = tile.astype(np.float32)
    lo = np.percentile(tile, q_low, axis=(-2, -1), keepdims=True)
    hi = np.percentile(tile, q_high, axis=(-2, -1), keepdims=True)
    return (tile - lo) / np.maximum(hi - lo, 1e-6)


def _soft_weight(h, w, edge_fraction=0.04, min_weight=0.85):
    yy = np.ones(h, dtype=np.float32)
    xx = np.ones(w, dtype=np.float32)
    ey = max(1, int(h * edge_fraction))
    ex = max(1, int(w * edge_fraction))
    ry = np.linspace(min_weight, 1.0, ey, dtype=np.float32)
    rx = np.linspace(min_weight, 1.0, ex, dtype=np.float32)
    yy[:ey] = ry
    yy[-ey:] = ry[::-1]
    xx[:ex] = rx
    xx[-ex:] = rx[::-1]
    return np.outer(yy, xx)


def stitch_metadata_clean(czi_path, tiles, scene=0, time=0, z=0, channel_ref=0, crop_edge=10, normalize=True, offset_x=0, offset_y=0, edge_fraction=0.04, min_weight=0.85):
    """Stitch ``(M, C, Y, X)`` tiles using CZI mosaic bounding boxes."""
    czi = CziFile(czi_path)
    M, C, Y, X = tiles.shape
    dtype = tiles.dtype
    positions = []
    for m in range(M):
        bb = czi.get_mosaic_tile_bounding_box(M=m, S=scene, T=time, C=channel_ref, Z=z)
        positions.append({"M": m, "x": int(bb.x), "y": int(bb.y), "w": int(bb.w), "h": int(bb.h)})

    xs = sorted(set(p["x"] for p in positions))
    ys = sorted(set(p["y"] for p in positions))
    for p in positions:
        p["col"] = int(np.argmin([abs(p["x"] - x) for x in xs]))
        p["row"] = int(np.argmin([abs(p["y"] - y) for y in ys]))
        p["x_adj"] = p["x"] + p["col"] * offset_x
        p["y_adj"] = p["y"] + p["row"] * offset_y

    min_x = min(p["x_adj"] for p in positions)
    min_y = min(p["y_adj"] for p in positions)
    for p in positions:
        p["x0"] = int(p["x_adj"] - min_x)
        p["y0"] = int(p["y_adj"] - min_y)

    ce = int(crop_edge)
    tile_h = Y - 2 * ce if ce else Y
    tile_w = X - 2 * ce if ce else X
    max_x = max(p["x0"] + ce + tile_w for p in positions)
    max_y = max(p["y0"] + ce + tile_h for p in positions)
    canvas = np.zeros((C, max_y, max_x), dtype=np.float32)
    weights = np.zeros((1, max_y, max_x), dtype=np.float32)

    for p in positions:
        tile = tiles[p["M"]].astype(np.float32)
        if normalize:
            scale = np.percentile(tile, 99.8, axis=(-2, -1), keepdims=True)
            tile = _normalize_tile(tile) * scale
        tile = tile[:, ce:Y - ce if ce else Y, ce:X - ce if ce else X]
        h, w = tile.shape[-2:]
        y = p["y0"] + ce
        x = p["x0"] + ce
        weight = _soft_weight(h, w, edge_fraction=edge_fraction, min_weight=min_weight)[None, :, :]
        canvas[:, y:y + h, x:x + w] += tile * weight
        weights[:, y:y + h, x:x + w] += weight

    stitched = canvas / np.maximum(weights, 1e-6)
    if np.issubdtype(dtype, np.integer):
        stitched = np.clip(stitched, 0, np.iinfo(dtype).max)
    return stitched.astype(dtype), positions


def _dims_dict(czi):
    return czi.get_dims_shape()[0]


def _axis_size(dims, axis, default=1):
    return int(dims.get(axis, (0, default))[1])


def _read_plane(czi, scene=0, time=0, channel=0, z=0, mosaic=None):
    kwargs = {"S": scene, "T": time, "C": channel, "Z": z}
    if mosaic is not None:
        kwargs["M"] = mosaic
    arr, _ = czi.read_image(**kwargs)
    arr = np.squeeze(arr)
    if arr.ndim > 2:
        arr = arr.reshape((-1,) + arr.shape[-2:])[0]
    return arr


def read_frame_cyx(czi_path, scene=0, time=0, z=0, channel_indices=None, stitch=True, stitch_kwargs=None):
    """Read one time point from a CZI as ``(C, Y, X)``."""
    czi = CziFile(czi_path)
    dims = _dims_dict(czi)
    n_channels = _axis_size(dims, "C")
    n_mosaic = _axis_size(dims, "M")
    channel_indices = list(range(n_channels)) if channel_indices is None else list(channel_indices)

    if stitch and n_mosaic > 1:
        tiles = []
        for m in range(n_mosaic):
            per_channel = [_read_plane(czi, scene, time, c, z, mosaic=m) for c in channel_indices]
            tiles.append(np.stack(per_channel, axis=0))
        tiles = np.stack(tiles, axis=0)
        stitched, positions = stitch_metadata_clean(
            czi_path=czi_path,
            tiles=tiles,
            scene=scene,
            time=time,
            z=z,
            channel_ref=channel_indices[0],
            **(stitch_kwargs or {}),
        )
        return stitched, positions

    frame = np.stack([_read_plane(czi, scene, time, c, z, mosaic=None) for c in channel_indices], axis=0)
    return frame, []


def _write_channel_tiff(path, image, frame_id, channel_name, metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(
        path,
        image,
        metadata={
            "axes": "YX",
            "frame_id": int(frame_id),
            "channel_name": str(channel_name),
            **metadata,
        },
    )
    return str(path)


def build_metadata(channel_names, image_shape, pixel_size_x, pixel_size_y, pixel_unit="um", origin_x=0.0, origin_y=0.0):
    return {
        "channels": list(channel_names),
        "size": {"C": len(channel_names), "Y": int(image_shape[-2]), "X": int(image_shape[-1])},
        "physical_pixel_size": {"X": float(pixel_size_x), "Y": float(pixel_size_y), "X_unit": pixel_unit, "Y_unit": pixel_unit},
        "origin": {"X": float(origin_x), "Y": float(origin_y), "unit": pixel_unit},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--czi", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--scene", type=int, default=0)
    parser.add_argument("--z", type=int, default=0)
    parser.add_argument("--time-start", type=int, default=0)
    parser.add_argument("--time-end", type=int, default=None, help="Exclusive. Defaults to all CZI time points.")
    parser.add_argument("--channel-names", default=None, help="Comma-separated names. Defaults to Channel-0,...")
    parser.add_argument("--channel-indices", default=None, help="Comma-separated CZI channel indices. Defaults to all.")
    parser.add_argument("--segment-channel", default="Cy5")
    parser.add_argument("--model-info", default="2D_versatile_fluo")
    parser.add_argument("--prob-thresh", type=float, default=0.45)
    parser.add_argument("--nms-thresh", type=float, default=0.2)
    parser.add_argument("--pixel-size-x", type=float, default=1.0)
    parser.add_argument("--pixel-size-y", type=float, default=1.0)
    parser.add_argument("--pixel-unit", default="um")
    parser.add_argument("--origin-x", type=float, default=0.0)
    parser.add_argument("--origin-y", type=float, default=0.0)
    parser.add_argument("--no-stitch", action="store_true")
    parser.add_argument("--crop-edge", type=int, default=10)
    parser.add_argument("--no-stitch-normalize", action="store_true")
    parser.add_argument("--offset-x", type=int, default=0)
    parser.add_argument("--offset-y", type=int, default=0)
    args = parser.parse_args()

    czi_path = Path(args.czi)
    out_dir = Path(args.out_dir)
    czi = CziFile(czi_path)
    dims = _dims_dict(czi)
    n_times = _axis_size(dims, "T")
    n_channels = _axis_size(dims, "C")
    channel_indices = list(range(n_channels)) if args.channel_indices is None else [int(x) for x in args.channel_indices.split(",")]
    channel_names = (
        [f"Channel-{i}" for i in channel_indices]
        if args.channel_names is None
        else [x.strip() for x in args.channel_names.split(",") if x.strip()]
    )
    if len(channel_names) != len(channel_indices):
        raise ValueError("--channel-names must have one name per selected channel.")

    time_end = n_times if args.time_end is None else args.time_end
    rows = []
    meta = None
    stitch_kwargs = {
        "crop_edge": args.crop_edge,
        "normalize": not args.no_stitch_normalize,
        "offset_x": args.offset_x,
        "offset_y": args.offset_y,
    }

    for t in range(args.time_start, time_end):
        frame, positions = read_frame_cyx(
            czi_path,
            scene=args.scene,
            time=t,
            z=args.z,
            channel_indices=channel_indices,
            stitch=not args.no_stitch,
            stitch_kwargs=stitch_kwargs,
        )
        meta = build_metadata(channel_names, frame.shape, args.pixel_size_x, args.pixel_size_y, args.pixel_unit, args.origin_x, args.origin_y)

        frame_id = int(t)
        row = {"frame_id": frame_id, "scene": int(args.scene), "z": int(args.z)}
        channel_files = []
        for local_c, channel_name in enumerate(channel_names):
            channel_file = out_dir / "stitched_channels" / channel_name / f"frame_{frame_id:06d}_{channel_name}.tif"
            channel_files.append(_write_channel_tiff(channel_file, frame[local_c], frame_id, channel_name, {"source_czi": str(czi_path)}))
            row[f"channel_{local_c}_name"] = channel_name
            row[f"channel_{local_c}_file"] = channel_files[-1]

        labels, _, seg_name = segment_image_to_labels(
            frame,
            meta,
            stardist_model=args.model_info,
            segment_channel=args.segment_channel,
            prob_thresh=args.prob_thresh,
            nms_thresh=args.nms_thresh,
        )
        mask_file = out_dir / "segmentation_masks" / seg_name / f"frame_{frame_id:06d}_{seg_name}_labels.tif"
        row["mask_file"] = write_label_tiff(mask_file, labels, meta, frame_id=frame_id, seg_channel_name=seg_name)
        row["seg_channel_name"] = seg_name
        row["n_labels"] = int(labels.max()) if labels.size else 0
        rows.append(row)

        if positions:
            pos_file = out_dir / "stitch_positions" / f"frame_{frame_id:06d}_positions.json"
            pos_file.parent.mkdir(parents=True, exist_ok=True)
            pos_file.write_text(json.dumps(positions, indent=2))

    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.DataFrame(rows)
    manifest_path = out_dir / "tiff_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote manifest: {manifest_path}")
    print(f"Wrote metadata: {out_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
