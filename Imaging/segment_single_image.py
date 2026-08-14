#!/usr/bin/env python3
"""Standalone extraction of the repo's single-image segmentation logic.

This preserves the segmentation behavior from
``pipelines/segmentation/scripts/segment.py`` for one time point:

- StarDist 2D pretrained model ``2D_versatile_fluo``
- segmentation on one selected channel only (pipeline config: ``Cy5``)
- intensity summaries added for every channel in the output CSV
- the same region-properties and physical-centroid calculations

Expected input image shape is either your native ``(C, Y, X)`` or the repo-style
``(C, Y, X, 3)`` for one time point.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from csbdeep.utils import normalize
from skimage.color import rgb2gray
from skimage.measure import regionprops, regionprops_table
from stardist.models import StarDist2D


def _parse_channel_list(value):
    """Parse CLI channel/radius lists while preserving the original "all" behavior."""
    if value == "all" or isinstance(value, (list, tuple)):
        return value
    if value is None or value == "":
        return []
    return [v.strip() for v in str(value).split(",") if v.strip()]


def _as_gray(channel_image):
    """Return a 2D grayscale channel image, matching repo behavior for RGB-like input."""
    channel_image = np.asarray(channel_image)
    if channel_image.ndim == 2:
        return channel_image
    return rgb2gray(channel_image)


def _scale_to_uint8(values):
    """Robustly scale intensities to uint8 for texture/crop descriptors."""
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.zeros(arr.shape, dtype=np.uint8)
    lo, hi = np.nanpercentile(arr[finite], [1, 99])
    if hi <= lo:
        lo, hi = np.nanmin(arr[finite]), np.nanmax(arr[finite])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    arr = np.clip((arr - lo) / (hi - lo), 0, 1)
    return (arr * 255).astype(np.uint8)


def _crop_centered(image_2d, cy, cx, size=32, fill=0):
    """Return a fixed-size crop centered on ``(cy, cx)`` with padding as needed."""
    image_2d = np.asarray(image_2d)
    half = size // 2
    y0, y1 = int(round(cy)) - half, int(round(cy)) - half + size
    x0, x1 = int(round(cx)) - half, int(round(cx)) - half + size

    crop = np.full((size, size), fill, dtype=image_2d.dtype)
    src_y0, src_y1 = max(0, y0), min(image_2d.shape[0], y1)
    src_x0, src_x1 = max(0, x0), min(image_2d.shape[1], x1)
    dst_y0, dst_x0 = src_y0 - y0, src_x0 - x0
    crop[dst_y0:dst_y0 + (src_y1 - src_y0), dst_x0:dst_x0 + (src_x1 - src_x0)] = image_2d[src_y0:src_y1, src_x0:src_x1]
    return crop


def _write_crops_npz(out_crops, image, df, info, crop_size=64, crop_channels="all"):
    """Save fixed-size per-cell crops to a compressed .npz file and return df with crop metadata.

    The output array has shape ``(N_cells, N_channels, crop_size, crop_size)``.
    Each channel is stored as a 2D grayscale plane, so native CYX and repo-style
    CYX3 inputs produce the same crop representation.
    """
    if out_crops is None or df.empty:
        return df

    ch_names = info.get("channels", [])

    def _to_idx(ch):
        if isinstance(ch, str):
            stripped = ch.strip()
            try:
                return int(stripped)
            except ValueError:
                return ch_names.index(stripped)
        return int(ch)

    if crop_channels == "all":
        channel_idx = list(range(image.shape[0]))
    else:
        channel_idx = [_to_idx(c) for c in _parse_channel_list(crop_channels)]

    crops = []
    crop_indices = []
    crop_ids = []
    labels_out = []
    centroid_y = []
    centroid_x = []

    for crop_index, (_, row) in enumerate(df.iterrows()):
        cy = float(row["centroid_y_px"])
        cx = float(row["centroid_x_px"])
        per_channel = []
        for c in channel_idx:
            raw_c = _as_gray(image[c])
            per_channel.append(_crop_centered(raw_c, cy, cx, size=crop_size))

        crops.append(np.stack(per_channel, axis=0))
        crop_indices.append(crop_index)
        label = int(row["label"])
        labels_out.append(label)
        centroid_y.append(cy)
        centroid_x.append(cx)
        crop_ids.append(f"cell_{label:06d}_crop_{crop_index:06d}")

    out_crops = Path(out_crops)
    out_crops.parent.mkdir(parents=True, exist_ok=True)

    crops = np.stack(crops, axis=0) if crops else np.empty((0, len(channel_idx), crop_size, crop_size))
    np.savez_compressed(
        out_crops,
        crops=crops,
        crop_index=np.asarray(crop_indices, dtype=np.int64),
        crop_id=np.asarray(crop_ids),
        label=np.asarray(labels_out, dtype=np.int64),
        centroid_y_px=np.asarray(centroid_y, dtype=float),
        centroid_x_px=np.asarray(centroid_x, dtype=float),
        channel_index=np.asarray(channel_idx, dtype=np.int64),
        channel_names=np.asarray([ch_names[c] if c < len(ch_names) else f"Channel-{c}" for c in channel_idx]),
        crop_size=np.asarray(crop_size, dtype=np.int64),
    )

    df = df.copy()
    df["crop_index"] = crop_indices
    df["crop_id"] = crop_ids
    df["crop_file"] = str(out_crops)
    return df


def _append_crops_zarr(out_zarr, image, df, info, frame_id=0, crop_size=64, crop_channels="all", chunks=(256, 1, 64, 64)):
    """Append fixed-size per-cell crops to a Zarr store and return df with crop metadata.

    The primary dataset is ``/crops`` with shape
    ``(N_total_cells, N_channels, crop_size, crop_size)``. Calling this function
    repeatedly with new frames appends along axis 0.
    """
    if out_zarr is None or df.empty:
        return df

    try:
        import zarr
    except Exception as exc:
        raise ImportError("Writing crops to Zarr requires `pip install zarr`.") from exc

    ch_names = info.get("channels", [])

    def _to_idx(ch):
        if isinstance(ch, str):
            stripped = ch.strip()
            try:
                return int(stripped)
            except ValueError:
                return ch_names.index(stripped)
        return int(ch)

    if crop_channels == "all":
        channel_idx = list(range(image.shape[0]))
    else:
        channel_idx = [_to_idx(c) for c in _parse_channel_list(crop_channels)]
    channel_names = [ch_names[c] if c < len(ch_names) else f"Channel-{c}" for c in channel_idx]

    frame_crops = []
    labels_out = []
    centroid_y = []
    centroid_x = []
    crop_ids = []

    for local_crop_index, (_, row) in enumerate(df.iterrows()):
        cy = float(row["centroid_y_px"])
        cx = float(row["centroid_x_px"])
        per_channel = []
        for c in channel_idx:
            raw_c = _as_gray(image[c])
            per_channel.append(_crop_centered(raw_c, cy, cx, size=crop_size))

        frame_crops.append(np.stack(per_channel, axis=0))
        label = int(row["label"])
        labels_out.append(label)
        centroid_y.append(cy)
        centroid_x.append(cx)
        crop_ids.append(f"frame_{int(frame_id):06d}_cell_{label:06d}_crop_{local_crop_index:06d}")

    frame_crops = np.stack(frame_crops, axis=0) if frame_crops else np.empty((0, len(channel_idx), crop_size, crop_size), dtype=image.dtype)
    n_new = int(frame_crops.shape[0])

    root = zarr.open_group(str(out_zarr), mode="a")

    # Store channel/crop metadata as attributes. Enforce consistency on append.
    if "channel_index" in root.attrs:
        existing_channel_index = list(root.attrs["channel_index"])
        existing_crop_size = int(root.attrs["crop_size"])
        if existing_channel_index != list(map(int, channel_idx)) or existing_crop_size != int(crop_size):
            raise ValueError(
                "Existing Zarr crop store has different crop_channels or crop_size. "
                "Use a new store or keep these settings fixed across frames."
            )
    else:
        root.attrs["channel_index"] = list(map(int, channel_idx))
        root.attrs["channel_names"] = list(map(str, channel_names))
        root.attrs["crop_size"] = int(crop_size)
        root.attrs["array_layout"] = "crops: (N_cells, N_channels, crop_size, crop_size)"

    crop_chunks = (min(int(chunks[0]), max(n_new, 1)), len(channel_idx), crop_size, crop_size)

    if "crops" not in root:
        crops_arr = root.create_dataset(
            "crops",
            shape=(0, len(channel_idx), crop_size, crop_size),
            chunks=crop_chunks,
            dtype=frame_crops.dtype,
        )
        frame_arr = root.create_dataset("frame_id", shape=(0,), chunks=(max(crop_chunks[0], 1),), dtype="i8")
        label_arr = root.create_dataset("label", shape=(0,), chunks=(max(crop_chunks[0], 1),), dtype="i8")
        cy_arr = root.create_dataset("centroid_y_px", shape=(0,), chunks=(max(crop_chunks[0], 1),), dtype="f8")
        cx_arr = root.create_dataset("centroid_x_px", shape=(0,), chunks=(max(crop_chunks[0], 1),), dtype="f8")
    else:
        crops_arr = root["crops"]
        frame_arr = root["frame_id"]
        label_arr = root["label"]
        cy_arr = root["centroid_y_px"]
        cx_arr = root["centroid_x_px"]

    start = int(crops_arr.shape[0])
    stop = start + n_new

    for arr, shape_tail in [
        (crops_arr, (len(channel_idx), crop_size, crop_size)),
        (frame_arr, ()),
        (label_arr, ()),
        (cy_arr, ()),
        (cx_arr, ()),
    ]:
        if hasattr(arr, "resize"):
            arr.resize((stop,) + shape_tail)
        else:
            raise RuntimeError("This Zarr array implementation does not support resize/append.")

    crops_arr[start:stop] = frame_crops
    frame_arr[start:stop] = np.full(n_new, int(frame_id), dtype=np.int64)
    label_arr[start:stop] = np.asarray(labels_out, dtype=np.int64)
    cy_arr[start:stop] = np.asarray(centroid_y, dtype=float)
    cx_arr[start:stop] = np.asarray(centroid_x, dtype=float)

    df = df.copy()
    df["frame_id"] = int(frame_id)
    df["zarr_index"] = np.arange(start, stop, dtype=np.int64)
    df["crop_index"] = df["zarr_index"]
    df["crop_id"] = crop_ids
    df["crop_store"] = str(out_zarr)
    df["crop_dataset"] = "crops"
    return df


def _boundary_fourier_features(label_mask, n_features=15):
    """Rotation-invariant shape features from normalized boundary Fourier magnitudes."""
    try:
        from skimage.segmentation import find_boundaries
    except Exception:
        return {f"shape_fft_{i:02d}": np.nan for i in range(n_features)}

    boundary = find_boundaries(label_mask, mode="inner")
    y, x = np.nonzero(boundary)
    if len(x) < 8:
        return {f"shape_fft_{i:02d}": np.nan for i in range(n_features)}

    cy, cx = y.mean(), x.mean()
    angles = np.arctan2(y - cy, x - cx)
    radii = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
    order = np.argsort(angles)
    angles = angles[order]
    radii = radii[order]

    target_angles = np.linspace(-np.pi, np.pi, 128, endpoint=False)
    # Close the periodic curve for interpolation.
    angles_ext = np.r_[angles, angles[0] + 2 * np.pi]
    radii_ext = np.r_[radii, radii[0]]
    target = target_angles.copy()
    target[target < angles_ext[0]] += 2 * np.pi
    sampled = np.interp(target, angles_ext, radii_ext)
    sampled = sampled - sampled.mean()

    fft_mag = np.abs(np.fft.rfft(sampled))
    norm = fft_mag[1] if len(fft_mag) > 1 and fft_mag[1] != 0 else (fft_mag.sum() or 1.0)
    vals = fft_mag[1:n_features + 1] / norm
    out = {f"shape_fft_{i:02d}": np.nan for i in range(n_features)}
    for i, v in enumerate(vals):
        out[f"shape_fft_{i:02d}"] = float(v)
    return out


def _mahotas_optional_features(intensity_crop, mask_crop, zernike_degree=12):
    """Compute optional Zernike and 13 Haralick features when mahotas is installed."""
    out = {}
    try:
        import mahotas as mh
    except Exception:
        for i in range(49):
            out[f"zernike_{i:02d}"] = np.nan
        for i in range(13):
            out[f"haralick_{i:02d}"] = np.nan
        out["optional_feature_backend"] = "mahotas_missing"
        return out

    mask_u8 = mask_crop.astype(bool)
    try:
        z = mh.features.zernike_moments(mask_u8.astype(np.uint8), radius=max(mask_crop.shape) // 2, degree=zernike_degree)
    except Exception:
        z = []
    for i in range(49):
        out[f"zernike_{i:02d}"] = float(z[i]) if i < len(z) else np.nan

    try:
        tex = _scale_to_uint8(intensity_crop)
        tex = np.where(mask_u8, tex, 0).astype(np.uint8)
        h = mh.features.haralick(tex, ignore_zeros=True).mean(axis=0)
    except Exception:
        h = []
    for i in range(13):
        out[f"haralick_{i:02d}"] = float(h[i]) if i < len(h) else np.nan
    out["optional_feature_backend"] = "mahotas"
    return out


def _glcm_fallback_features(intensity_crop, mask_crop):
    """Small set of GLCM texture features available from scikit-image."""
    out = {}
    try:
        try:
            from skimage.feature import graycomatrix, graycoprops
        except ImportError:
            from skimage.feature import greycomatrix as graycomatrix
            from skimage.feature import greycoprops as graycoprops

        tex = (_scale_to_uint8(intensity_crop) // 32).astype(np.uint8)
        tex = np.where(mask_crop.astype(bool), tex, 0).astype(np.uint8)
        glcm = graycomatrix(tex, distances=[1, 2, 4], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=8, symmetric=True, normed=True)
        for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
            vals = graycoprops(glcm, prop)
            out[f"glcm_{prop}_mean"] = float(np.nanmean(vals))
            out[f"glcm_{prop}_std"] = float(np.nanstd(vals))
    except Exception:
        for prop in ["contrast", "dissimilarity", "homogeneity", "ASM", "energy", "correlation"]:
            out[f"glcm_{prop}_mean"] = np.nan
            out[f"glcm_{prop}_std"] = np.nan
    return out


def _add_oblique_tile_features(feats, image, labels, label, cy, cx, oblique_idx, oblique_tile_size=34):
    """Append features computed from a centered oblique-channel tile.

    The oblique intensity tile is centered on the segmented object's centroid.
    The object mask tile is cropped from the existing segmentation mask; this
    keeps object localization consistent with the StarDist segmentation while
    using the oblique channel for intensity and texture measurements.
    """
    if oblique_idx is None:
        return feats

    raw_oblique = _as_gray(image[oblique_idx])
    oblique_tile = _crop_centered(raw_oblique, cy, cx, size=oblique_tile_size)
    mask_tile = _crop_centered(labels == label, cy, cx, size=oblique_tile_size).astype(bool)

    vals_tile = oblique_tile[np.isfinite(oblique_tile)]
    vals_masked = oblique_tile[mask_tile]

    feats["oblique_tile_size"] = int(oblique_tile_size)
    feats["oblique_tile_mask_area_px"] = int(mask_tile.sum())

    if vals_tile.size:
        feats["oblique_tile_mean"] = float(np.mean(vals_tile))
        feats["oblique_tile_std"] = float(np.std(vals_tile))
        feats["oblique_tile_min"] = float(np.min(vals_tile))
        feats["oblique_tile_max"] = float(np.max(vals_tile))
        feats["oblique_tile_p05"] = float(np.percentile(vals_tile, 5))
        feats["oblique_tile_p50"] = float(np.percentile(vals_tile, 50))
        feats["oblique_tile_p95"] = float(np.percentile(vals_tile, 95))
    else:
        for name in ["mean", "std", "min", "max", "p05", "p50", "p95"]:
            feats[f"oblique_tile_{name}"] = np.nan

    if vals_masked.size:
        feats["oblique_masked_mean"] = float(np.mean(vals_masked))
        feats["oblique_masked_std"] = float(np.std(vals_masked))
        feats["oblique_masked_min"] = float(np.min(vals_masked))
        feats["oblique_masked_max"] = float(np.max(vals_masked))
        feats["oblique_masked_sum"] = float(np.sum(vals_masked))
        feats["oblique_masked_p05"] = float(np.percentile(vals_masked, 5))
        feats["oblique_masked_p50"] = float(np.percentile(vals_masked, 50))
        feats["oblique_masked_p95"] = float(np.percentile(vals_masked, 95))
    else:
        for name in ["mean", "std", "min", "max", "sum", "p05", "p50", "p95"]:
            feats[f"oblique_masked_{name}"] = np.nan

    # Texture features on the oblique tile. The fallback GLCM features are computed
    # both for the full tile and for the segmentation-masked tile.
    for k, v in _glcm_fallback_features(oblique_tile, np.ones_like(mask_tile, dtype=bool)).items():
        feats[f"oblique_tile_{k}"] = v
    for k, v in _glcm_fallback_features(oblique_tile, mask_tile).items():
        feats[f"oblique_masked_{k}"] = v

    # Optional Haralick/Zernike backend. Haralick uses oblique intensity;
    # Zernike uses the object mask inside the oblique tile.
    for k, v in _mahotas_optional_features(oblique_tile, mask_tile).items():
        feats[f"oblique_{k}"] = v

    # Shape Fourier descriptors for the object mask clipped to the oblique tile.
    # These describe the segmented object footprint visible in the oblique tile.
    for k, v in _boundary_fourier_features(mask_tile, n_features=15).items():
        feats[f"oblique_tile_{k}"] = v

    return feats


def _add_extended_features(df, labels, image, info, seg_idx, sum_idx, brightfield_channel=None, crop_size=32, neighborhood_radii=(50, 100), include_raw_crops=False, oblique_channel=None, oblique_tile_size=34):
    """Add downstream-analysis features that are valid for a single segmented frame."""
    if df.empty:
        return df

    ch_names = info.get("channels", [])

    def _to_idx(ch):
        if ch is None:
            return None
        if isinstance(ch, str):
            stripped = ch.strip()
            try:
                return int(stripped)
            except ValueError:
                return ch_names.index(stripped)
        return int(ch)

    raw_seg = _as_gray(image[seg_idx])
    bf_idx = _to_idx(brightfield_channel) if brightfield_channel is not None else None
    raw_bf = _as_gray(image[bf_idx]) if bf_idx is not None else raw_seg
    oblique_idx = _to_idx(oblique_channel) if oblique_channel is not None else None

    prop_by_label = {p.label: p for p in regionprops(labels, intensity_image=raw_seg)}
    rows = []
    for _, base in df.iterrows():
        label = int(base["label"])
        prop = prop_by_label.get(label)
        if prop is None:
            rows.append({"label": label})
            continue

        minr, minc, maxr, maxc = prop.bbox
        obj_mask = labels[minr:maxr, minc:maxc] == label
        seg_crop = raw_seg[minr:maxr, minc:maxc]
        fixed_seg_crop = _crop_centered(raw_seg, base["centroid_y_px"], base["centroid_x_px"], crop_size)
        fixed_mask_crop = _crop_centered(labels == label, base["centroid_y_px"], base["centroid_x_px"], crop_size).astype(bool)
        fixed_bf_crop = _crop_centered(raw_bf, base["centroid_y_px"], base["centroid_x_px"], crop_size)

        feats = {"label": label}
        # Additional direct shape descriptors.
        feats["area_um2"] = float(base["area"]) * float(info["physical_pixel_size"]["X"]) * float(info["physical_pixel_size"]["Y"])
        feats["aspect_ratio"] = float(prop.major_axis_length / prop.minor_axis_length) if prop.minor_axis_length else np.nan
        feats["circularity"] = float(4 * np.pi * prop.area / (prop.perimeter ** 2)) if prop.perimeter else np.nan
        feats["compactness"] = float((prop.perimeter ** 2) / prop.area) if prop.area else np.nan
        feats["bbox_area_fraction"] = float(prop.area / ((maxr - minr) * (maxc - minc))) if (maxr > minr and maxc > minc) else np.nan

        # Texture / morphology descriptors used in morphodynamic-style pipelines.
        feats.update(_mahotas_optional_features(seg_crop, obj_mask))
        feats.update(_glcm_fallback_features(seg_crop, obj_mask))
        feats.update(_boundary_fourier_features(labels == label, n_features=15))

        if oblique_idx is not None:
            feats = _add_oblique_tile_features(
                feats=feats,
                image=image,
                labels=labels,
                label=label,
                cy=float(base["centroid_y_px"]),
                cx=float(base["centroid_x_px"]),
                oblique_idx=oblique_idx,
                oblique_tile_size=oblique_tile_size,
            )

        # Per-channel robust intensity percentiles inside the object.
        for c in sum_idx:
            raw_c = _as_gray(image[c])
            ch_label = ch_names[c] if c < len(ch_names) else f"Channel-{c}"
            vals = raw_c[labels == label]
            if vals.size:
                feats[f"{ch_label}_p05"] = float(np.percentile(vals, 5))
                feats[f"{ch_label}_p50"] = float(np.percentile(vals, 50))
                feats[f"{ch_label}_p95"] = float(np.percentile(vals, 95))
                feats[f"{ch_label}_std"] = float(np.std(vals))
            else:
                feats[f"{ch_label}_p05"] = feats[f"{ch_label}_p50"] = feats[f"{ch_label}_p95"] = feats[f"{ch_label}_std"] = np.nan

        # Imaging-flow-cytometry-style fixed raw brightfield crop summaries.
        bf_u8 = _scale_to_uint8(fixed_bf_crop)
        feats["ifc_crop_channel"] = ch_names[bf_idx] if bf_idx is not None and bf_idx < len(ch_names) else "segmentation_channel"
        feats["ifc_crop_size"] = crop_size
        feats["ifc_crop_mean"] = float(np.mean(bf_u8))
        feats["ifc_crop_std"] = float(np.std(bf_u8))
        if include_raw_crops:
            flat = bf_u8.reshape(-1)
            for i, v in enumerate(flat):
                feats[f"ifc_crop_px_{i:04d}"] = int(v)

        rows.append(feats)

    extra = pd.DataFrame(rows)

    # Neighborhood/contact descriptors: single-frame environment features.
    coords = df[["label", "centroid_y_px", "centroid_x_px", "area"]].copy()
    yx = coords[["centroid_y_px", "centroid_x_px"]].to_numpy(float)
    env_rows = []
    for i, row in coords.iterrows():
        d = np.sqrt(((yx - np.array([row["centroid_y_px"], row["centroid_x_px"]])) ** 2).sum(axis=1))
        d[d == 0] = np.nan
        env = {"label": int(row["label"]), "nearest_neighbor_dist_px": float(np.nanmin(d)) if np.isfinite(d).any() else np.nan}
        for r in neighborhood_radii:
            r = float(r)
            env[f"neighbors_within_{int(r)}px"] = int(np.nansum(d <= r))
            env[f"local_density_{int(r)}px"] = float(np.nansum(d <= r) / (np.pi * r * r)) if r > 0 else np.nan
        env_rows.append(env)
    env_df = pd.DataFrame(env_rows)

    out = df.merge(extra, on="label", how="left").merge(env_df, on="label", how="left")
    out["motility_requires_tracking"] = True
    out["predicted_molecular_features_require_trained_model"] = True
    return out


def resolve_channel_index(segment_channel, meta):
    """Resolve a channel specification to an integer index."""
    chs = meta.get("channels") or []
    C = meta.get("size", {}).get("C", len(chs) or 1)

    if not segment_channel or str(segment_channel).lower() in {"auto", "none"}:
        return 0

    s = str(segment_channel).strip()

    try:
        idx = int(s)
        if idx < 0:
            idx += C
        return max(0, min(idx, C - 1))
    except ValueError:
        pass

    try:
        return [c.lower() for c in chs].index(s.lower())
    except ValueError:
        return 0


def _probe_supported_props(labels, intensity_image=None):
    """Return regionprops_table properties supported by this scikit-image build."""
    candidates = [
        "label", "area", "bbox", "bbox_area", "centroid", "eccentricity",
        "equivalent_diameter", "euler_number", "extent", "feret_diameter_max",
        "filled_area", "inertia_tensor_eigvals", "local_centroid",
        "major_axis_length", "minor_axis_length", "orientation",
        "perimeter", "perimeter_crofton", "solidity",
    ]
    supported = []
    for prop in candidates:
        try:
            regionprops_table(labels, intensity_image=intensity_image, properties=[prop])
            supported.append(prop)
        except Exception:
            pass
    return supported


def segment_single_image_to_csv(
    image,
    info,
    out_csv,
    stardist_model="2D_versatile_fluo",
    segment_channel="Cy5",
    prob_thresh=0.45,
    nms_thresh=0.2,
    summarize_channels="all",
    extract_extended_features=True,
    brightfield_channel=None,
    oblique_channel=None,
    oblique_tile_size=34,
    crop_size=32,
    neighborhood_radii=(50, 100),
    include_raw_crops=False,
    out_crops=None,
    out_zarr=None,
    crop_channels="all",
    frame_id=0,
):
    """Segment one multi-channel image and save the repo-equivalent props CSV.

    Parameters
    ----------
    image : np.ndarray
        Single time point with shape ``(C, Y, X)`` or ``(C, Y, X, 3)``.
    info : dict
        Metadata dictionary in the same structure produced by the repo's
        ``get_metadata.py`` script.
    out_csv : str or pathlib.Path
        Destination for the region-properties CSV.
    extract_extended_features : bool
        If true, append morphology, texture, boundary-Fourier, neighborhood,
        robust per-channel intensity, and fixed-crop summary features.
    out_crops : str or pathlib.Path, optional
        If provided, save raw fixed-size per-cell image crops to this compressed
        ``.npz`` file and add ``crop_index``, ``crop_id``, and ``crop_file``
        columns to the CSV. Useful for small tests.
    out_zarr : str or pathlib.Path, optional
        If provided, append raw fixed-size per-cell image crops to this Zarr
        store and add ``zarr_index``/``crop_store`` metadata columns to the CSV.
        Prefer this for many frames.
    crop_channels : "all" or sequence of channel names/indices
        Channels to store in crop outputs.
    frame_id : int
        Frame/time index to annotate CSV rows and Zarr crop metadata.
    oblique_channel : int or str, optional
        Channel index/name for oblique-channel tile features. If omitted, no
        oblique-specific feature block is added.
    oblique_tile_size : int
        Centered tile size for oblique features. Default is 34.

    Returns
    -------
    tuple[np.ndarray, pandas.DataFrame]
        ``labels`` with shape ``(Y, X)`` and the region-properties table.
    """
    image = np.asarray(image)
    if image.ndim == 3:
        # User-convenience for native grayscale channel stacks: (C, Y, X).
        # _as_gray() will then use each 2D channel directly, preserving the same signal
        # the repo would get after CYX -> CYX3 -> rgb2gray.
        pass
    elif image.ndim != 4:
        raise ValueError(f"Expected image shape (C, Y, X) or (C, Y, X, 3); got {image.shape}")

    ch_names = info.get("channels", [])

    def _to_idx(ch):
        if isinstance(ch, str):
            stripped = ch.strip()
            try:
                return int(stripped)
            except ValueError:
                return ch_names.index(stripped)
        return int(ch)

    seg_idx = resolve_channel_index(segment_channel, info)
    seg_name = ch_names[seg_idx] if seg_idx < len(ch_names) else f"Channel-{seg_idx}"

    if summarize_channels == "all":
        sum_idx = list(range(image.shape[0]))
    else:
        summarize_channels = _parse_channel_list(summarize_channels)
        sum_idx = [_to_idx(c) for c in summarize_channels]
    if seg_idx not in sum_idx:
        sum_idx = [seg_idx] + sum_idx

    sx = float(info["physical_pixel_size"]["X"])
    sy = float(info["physical_pixel_size"]["Y"])
    ox = float(info["origin"]["X"])
    oy = float(info["origin"]["Y"])
    unit_px = info["physical_pixel_size"].get("X_unit", "µm")
    unit_org = info["origin"].get("unit", "µm")
    if unit_px != unit_org:
        raise ValueError("Pixel-size and origin units differ; convert first.")

    C = image.shape[0]
    if not (0 <= seg_idx < C):
        raise IndexError(f"Channel index {seg_idx} out of range [0, {C - 1}]")

    model = StarDist2D.from_pretrained(stardist_model)

    raw_seg = _as_gray(image[seg_idx])
    labels, _ = model.predict_instances(
        normalize(raw_seg),
        prob_thresh=prob_thresh,
        nms_thresh=nms_thresh,
    )

    supported_props = _probe_supported_props(labels, intensity_image=raw_seg)
    main_props = regionprops_table(
        labels,
        intensity_image=raw_seg,
        properties=supported_props + ["mean_intensity", "max_intensity", "min_intensity"],
    )
    df = pd.DataFrame(main_props)

    if not df.empty:
        df["centroid_x_px"] = df["centroid-1"]
        df["centroid_y_px"] = df["centroid-0"]
        df["centroid_x_um"] = ox + df["centroid_x_px"] * sx
        df["centroid_y_um"] = oy + df["centroid_y_px"] * sy
        df["centroid_unit"] = unit_org

        df.rename(
            columns={
                "mean_intensity": f"{seg_name}_mean",
                "max_intensity": f"{seg_name}_max",
                "min_intensity": f"{seg_name}_min",
            },
            inplace=True,
        )
        df[f"{seg_name}_sum"] = df[f"{seg_name}_mean"] * df["area"]

        for c in sum_idx:
            if c == seg_idx:
                continue
            raw_c = _as_gray(image[c])
            ch_label = ch_names[c] if c < len(ch_names) else f"Channel-{c}"
            add = regionprops_table(
                labels,
                intensity_image=raw_c,
                properties=["label", "mean_intensity", "max_intensity", "min_intensity"],
            )
            add = pd.DataFrame(add).rename(
                columns={
                    "mean_intensity": f"{ch_label}_mean",
                    "max_intensity": f"{ch_label}_max",
                    "min_intensity": f"{ch_label}_min",
                }
            )
            add[f"{ch_label}_sum"] = add[f"{ch_label}_mean"] * df["area"]
            df = df.merge(add, on="label", how="left")

        df["time"] = int(frame_id)
        df["frame_id"] = int(frame_id)
        df["seg_channel_idx"] = seg_idx
        df["seg_channel_name"] = seg_name

        if extract_extended_features:
            df = _add_extended_features(
                df=df,
                labels=labels,
                image=image,
                info=info,
                seg_idx=seg_idx,
                sum_idx=sum_idx,
                brightfield_channel=brightfield_channel,
                oblique_channel=oblique_channel,
                oblique_tile_size=oblique_tile_size,
                crop_size=crop_size,
                neighborhood_radii=neighborhood_radii,
                include_raw_crops=include_raw_crops,
            )

    if out_crops is not None:
        df = _write_crops_npz(
            out_crops=out_crops,
            image=image,
            df=df,
            info=info,
            crop_size=crop_size,
            crop_channels=crop_channels,
        )

    if out_zarr is not None:
        df = _append_crops_zarr(
            out_zarr=out_zarr,
            image=image,
            df=df,
            info=info,
            frame_id=frame_id,
            crop_size=crop_size,
            crop_channels=crop_channels,
        )

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return labels.astype(np.int32, copy=False), df


def _load_array(path):
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError("CLI input currently expects a .npy array with shape (C, Y, X) or (C, Y, X, 3).")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to .npy array shaped (C, Y, X) or (C, Y, X, 3)")
    parser.add_argument("--meta", required=True, help="Path to metadata JSON")
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--model-info", default="2D_versatile_fluo")
    parser.add_argument("--segment-channel", default="Cy5")
    parser.add_argument("--prob-thresh", type=float, default=0.45)
    parser.add_argument("--nms-thresh", type=float, default=0.2)
    parser.add_argument("--summarize-channels", default="all")
    parser.add_argument("--no-extended-features", action="store_true")
    parser.add_argument("--brightfield-channel", default=None, help="Name/index of brightfield channel for crop summary features")
    parser.add_argument("--oblique-channel", default=None, help="Name/index of oblique channel for 34x34 tile features")
    parser.add_argument("--oblique-tile-size", type=int, default=34)
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--neighborhood-radii", default="50,100", help="Comma-separated radii in pixels")
    parser.add_argument("--include-raw-crops", action="store_true", help="Add raw crop pixels as CSV columns; usually prefer --out-crops instead")
    parser.add_argument("--out-crops", default=None, help="Optional .npz file for raw per-cell crops")
    parser.add_argument("--out-zarr", default=None, help="Optional Zarr store for appendable raw per-cell crops")
    parser.add_argument("--crop-channels", default="all", help="Channels to save in crop outputs: all or comma-separated names/indices")
    parser.add_argument("--frame-id", type=int, default=0, help="Frame/time index for CSV and Zarr metadata")
    args = parser.parse_args()

    image = _load_array(args.image)
    with open(args.meta, "r") as fh:
        info = json.load(fh)

    segment_single_image_to_csv(
        image=image,
        info=info,
        out_csv=args.out_csv,
        stardist_model=args.model_info,
        segment_channel=args.segment_channel,
        prob_thresh=args.prob_thresh,
        nms_thresh=args.nms_thresh,
        summarize_channels=args.summarize_channels,
        extract_extended_features=not args.no_extended_features,
        brightfield_channel=args.brightfield_channel,
        oblique_channel=args.oblique_channel,
        oblique_tile_size=args.oblique_tile_size,
        crop_size=args.crop_size,
        neighborhood_radii=[float(r) for r in _parse_channel_list(args.neighborhood_radii)],
        include_raw_crops=args.include_raw_crops,
        out_crops=args.out_crops,
        out_zarr=args.out_zarr,
        crop_channels=args.crop_channels,
        frame_id=args.frame_id,
    )


if __name__ == "__main__":
    main()
