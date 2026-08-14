"""Tracking and temporal features for per-frame cell segmentations.

The functions in this module are deliberately additive: they consume the
per-cell data frames produced by :func:`segment_single_image_to_csv` and retain
every existing column.  The primary output remains one row per cell per frame,
augmented with persistent track/lineage identifiers and time-series features.

Tracking is based on centroid displacement and optional area consistency.  It
is intended as a transparent default for well sampled 2-D movies, not as a
replacement for manual validation or a specialized tracker on difficult data.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


DEFAULT_TEMPORAL_COLUMNS = (
    "area_um2", "area", "perimeter", "perimeter_crofton",
    "major_axis_length", "minor_axis_length", "equivalent_diameter",
    "feret_diameter_max", "eccentricity", "solidity", "circularity",
    "compactness", "aspect_ratio", "extent", "bbox_area_fraction",
)


def _safe_slope(t, y):
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    good = np.isfinite(t) & np.isfinite(y)
    if good.sum() < 2 or np.ptp(t[good]) == 0:
        return np.nan
    return float(np.polyfit(t[good], y[good], 1)[0])


def _safe_divide(numerator, denominator):
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    return np.divide(
        numerator, denominator,
        out=np.full(np.broadcast_shapes(numerator.shape, denominator.shape), np.nan),
        where=np.isfinite(denominator) & (denominator != 0),
    )


def _resolve_temporal_columns(df, columns=None):
    """Select biologically meaningful numeric columns, including all channels."""
    if columns is not None:
        return [c for c in columns if c in df and pd.api.types.is_numeric_dtype(df[c])]

    chosen = [c for c in DEFAULT_TEMPORAL_COLUMNS if c in df]
    suffixes = ("_mean", "_sum", "_std", "_p05", "_p50", "_p95")
    chosen.extend(c for c in df.columns if c.endswith(suffixes))
    # Preserve order while removing duplicates and metadata-like rolling inputs.
    return list(dict.fromkeys(c for c in chosen if pd.api.types.is_numeric_dtype(df[c])))


def _prepare_observations(observations, frame_interval=1.0, time_unit="frame", timestamps=None):
    df = observations.copy()
    required = {"frame_id", "label", "centroid_x_px", "centroid_y_px"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Observations are missing required columns: {missing}")
    if df.duplicated(["frame_id", "label"]).any():
        raise ValueError("Each (frame_id, label) pair must be unique.")

    df["frame_id"] = pd.to_numeric(df["frame_id"], errors="raise").astype(int)
    df["label"] = pd.to_numeric(df["label"], errors="raise").astype(int)
    frame0 = int(df["frame_id"].min()) if len(df) else 0

    if timestamps is not None:
        if isinstance(timestamps, Mapping):
            vals = df["frame_id"].map(timestamps)
        else:
            ordered_frames = sorted(df["frame_id"].unique())
            if len(timestamps) != len(ordered_frames):
                raise ValueError("timestamps must have one value per unique frame.")
            vals = df["frame_id"].map(dict(zip(ordered_frames, timestamps)))
        if vals.isna().any():
            raise ValueError("A timestamp is required for every frame.")
        if pd.api.types.is_datetime64_any_dtype(vals) or isinstance(vals.iloc[0], (pd.Timestamp, np.datetime64)):
            ts = pd.to_datetime(vals)
            df["timestamp"] = ts
            df["elapsed_time"] = (ts - ts.min()).dt.total_seconds() / 3600.0
            time_unit = "hour"
        else:
            df["elapsed_time"] = pd.to_numeric(vals, errors="raise").astype(float)
    elif "elapsed_time" not in df:
        df["elapsed_time"] = (df["frame_id"] - frame0) * float(frame_interval)
    else:
        df["elapsed_time"] = pd.to_numeric(df["elapsed_time"], errors="raise").astype(float)

    df["time_unit"] = str(time_unit)
    if "centroid_x_um" not in df:
        df["centroid_x_um"] = df["centroid_x_px"].astype(float)
    if "centroid_y_um" not in df:
        df["centroid_y_um"] = df["centroid_y_px"].astype(float)
    return df.sort_values(["frame_id", "label"]).reset_index(drop=True)


def link_cell_tracks(
    observations,
    max_displacement_um=25.0,
    max_gap_frames=1,
    area_ratio_range=(0.35, 2.8),
    area_cost_weight=0.25,
    detect_divisions=True,
    division_combined_area_ratio=(0.65, 1.6),
    division_max_child_parent_ratio=1.15,
):
    """Link frame-level objects into tracks and detect one-to-two divisions.

    A division ends the parent track and starts two daughter tracks.  Unmatched
    detections start new tracks.  Short gaps can be bridged, but observations are
    never fabricated or interpolated.
    """
    df = observations.copy().reset_index(drop=True)
    if df.empty:
        for c in ["track_id", "parent_track_id", "lineage_id", "generation"]:
            df[c] = pd.Series(dtype="Int64")
        return df, pd.DataFrame()

    area_col = "area_um2" if "area_um2" in df else ("area" if "area" in df else None)
    frames = sorted(df["frame_id"].unique())
    next_track = 1
    active = {}
    track_meta = {}
    assignments = {}
    match_meta = {}
    division_relations = []

    def start_track(row_idx, parent=None, confidence=np.nan, event="birth"):
        nonlocal next_track
        tid = next_track
        next_track += 1
        lineage = track_meta[parent]["lineage_id"] if parent is not None else tid
        generation = track_meta[parent]["generation"] + 1 if parent is not None else 0
        track_meta[tid] = {
            "parent_track_id": parent, "lineage_id": lineage,
            "generation": generation, "start_frame": int(df.at[row_idx, "frame_id"]),
        }
        assignments[row_idx] = tid
        match_meta[row_idx] = (event, confidence, np.nan, 0)
        active[tid] = row_idx
        return tid

    for frame_pos, frame in enumerate(frames):
        current = list(df.index[df["frame_id"] == frame])
        if frame_pos == 0:
            for idx in current:
                start_track(idx)
            continue

        candidates = {
            tid: idx for tid, idx in active.items()
            if int(frame) - int(df.at[idx, "frame_id"]) <= int(max_gap_frames) + 1
        }
        used_tracks, used_rows = set(), set()

        # Division evidence is strongest across adjacent frames. Resolve the best
        # non-overlapping parent/daughter triples before ordinary one-to-one links.
        division_candidates = []
        if detect_divisions and area_col and len(current) >= 2:
            for tid, pidx in candidates.items():
                if int(frame) - int(df.at[pidx, "frame_id"]) != 1:
                    continue
                px, py = float(df.at[pidx, "centroid_x_um"]), float(df.at[pidx, "centroid_y_um"])
                pa = float(df.at[pidx, area_col])
                if not np.isfinite(pa) or pa <= 0:
                    continue
                nearby = []
                for cidx in current:
                    d = np.hypot(float(df.at[cidx, "centroid_x_um"]) - px, float(df.at[cidx, "centroid_y_um"]) - py)
                    ca = float(df.at[cidx, area_col])
                    if d <= max_displacement_um and np.isfinite(ca) and ca / pa <= division_max_child_parent_ratio:
                        nearby.append((cidx, d, ca))
                for left, right in combinations(nearby, 2):
                    ratio = (left[2] + right[2]) / pa
                    if division_combined_area_ratio[0] <= ratio <= division_combined_area_ratio[1]:
                        asym = abs(left[2] - right[2]) / (left[2] + right[2])
                        score = (left[1] + right[1]) / (2 * max_displacement_um) + abs(np.log(ratio)) + 0.25 * asym
                        division_candidates.append((score, tid, pidx, left[0], right[0], ratio, asym))
            for score, tid, pidx, c1, c2, ratio, asym in sorted(division_candidates):
                if tid in used_tracks or c1 in used_rows or c2 in used_rows:
                    continue
                confidence = float(np.exp(-score))
                child_ids = [start_track(c1, tid, confidence, "division_child"), start_track(c2, tid, confidence, "division_child")]
                active.pop(tid, None)
                used_tracks.add(tid)
                used_rows.update((c1, c2))
                division_relations.append({
                    "parent_track_id": tid, "child_track_ids": child_ids,
                    "frame_id": int(frame), "elapsed_time": float(df.at[c1, "elapsed_time"]),
                    "confidence": confidence, "combined_area_ratio": ratio,
                    "daughter_area_asymmetry": asym,
                })

        remaining_tracks = [(tid, idx) for tid, idx in candidates.items() if tid not in used_tracks]
        remaining_rows = [idx for idx in current if idx not in used_rows]
        if remaining_tracks and remaining_rows:
            cost = np.full((len(remaining_tracks), len(remaining_rows)), np.inf)
            distances = np.full_like(cost, np.inf)
            for i, (tid, pidx) in enumerate(remaining_tracks):
                gap = int(frame) - int(df.at[pidx, "frame_id"])
                allowed_distance = max_displacement_um * max(1, gap)
                for j, cidx in enumerate(remaining_rows):
                    d = np.hypot(
                        float(df.at[cidx, "centroid_x_um"]) - float(df.at[pidx, "centroid_x_um"]),
                        float(df.at[cidx, "centroid_y_um"]) - float(df.at[pidx, "centroid_y_um"]),
                    )
                    ratio_penalty = 0.0
                    if area_col:
                        pa, ca = float(df.at[pidx, area_col]), float(df.at[cidx, area_col])
                        ratio = ca / pa if pa > 0 else np.nan
                        if not np.isfinite(ratio) or not (area_ratio_range[0] <= ratio <= area_ratio_range[1]):
                            continue
                        ratio_penalty = area_cost_weight * abs(np.log(ratio))
                    if d <= allowed_distance:
                        distances[i, j] = d
                        cost[i, j] = d / allowed_distance + ratio_penalty + 0.05 * (gap - 1)
            finite = np.isfinite(cost)
            if finite.any():
                safe_cost = np.where(finite, cost, 1e9)
                rr, cc = linear_sum_assignment(safe_cost)
                for i, j in zip(rr, cc):
                    if not finite[i, j]:
                        continue
                    tid, pidx = remaining_tracks[i]
                    cidx = remaining_rows[j]
                    gap = int(frame) - int(df.at[pidx, "frame_id"]) - 1
                    assignments[cidx] = tid
                    match_meta[cidx] = ("linked", float(np.exp(-cost[i, j])), float(distances[i, j]), gap)
                    active[tid] = cidx
                    used_tracks.add(tid)
                    used_rows.add(cidx)

        for idx in current:
            if idx not in assignments:
                start_track(idx)
        # Retire tracks that can no longer be linked.
        active = {
            tid: idx for tid, idx in active.items()
            if int(frame) - int(df.at[idx, "frame_id"]) <= int(max_gap_frames)
        }

    df["track_id"] = pd.array([assignments[i] for i in df.index], dtype="Int64")
    df["parent_track_id"] = pd.array([track_meta[assignments[i]]["parent_track_id"] for i in df.index], dtype="Int64")
    df["lineage_id"] = pd.array([track_meta[assignments[i]]["lineage_id"] for i in df.index], dtype="Int64")
    df["generation"] = pd.array([track_meta[assignments[i]]["generation"] for i in df.index], dtype="Int64")
    df["link_type"] = [match_meta[i][0] for i in df.index]
    df["tracking_confidence"] = [match_meta[i][1] for i in df.index]
    df["link_distance_um"] = [match_meta[i][2] for i in df.index]
    df["gap_frames"] = [match_meta[i][3] for i in df.index]
    df["is_interpolated"] = False
    return df, pd.DataFrame(division_relations)


def add_temporal_features(
    tracked,
    temporal_columns=None,
    rolling_windows=(3.0, 6.0, 12.0),
    msd_lags=(1, 2, 3, 5),
    stationary_speed_threshold=1.0,
):
    """Add causal rolling/expanding features to one-row-per-cell-frame data.

    Rolling windows and rates use ``elapsed_time`` and therefore support
    irregular sampling.  All rolling windows look backward, including the
    current observation, so they are safe for prospective prediction.
    """
    df = tracked.sort_values(["track_id", "elapsed_time", "frame_id"]).copy()
    temporal_columns = _resolve_temporal_columns(df, temporal_columns)

    motion_cols = [
        "delta_t", "dx_um", "dy_um", "step_distance_um", "speed_um_per_time",
        "velocity_x_um_per_time", "velocity_y_um_per_time", "acceleration_um_per_time2",
        "turning_angle_rad", "angular_velocity_rad_per_time", "distance_from_origin_um",
        "cumulative_path_length_um", "net_displacement_um", "directional_persistence",
    ]
    for col in motion_cols:
        df[col] = np.nan

    for tid, idx in df.groupby("track_id", sort=False).groups.items():
        idx = list(idx)
        t = df.loc[idx, "elapsed_time"].to_numpy(float)
        x = df.loc[idx, "centroid_x_um"].to_numpy(float)
        y = df.loc[idx, "centroid_y_um"].to_numpy(float)
        dt = np.r_[np.nan, np.diff(t)]
        dx, dy = np.r_[np.nan, np.diff(x)], np.r_[np.nan, np.diff(y)]
        step = np.hypot(dx, dy)
        speed, vx, vy = _safe_divide(step, dt), _safe_divide(dx, dt), _safe_divide(dy, dt)
        heading = np.arctan2(dy, dx)
        turn = np.r_[np.nan, np.diff(heading)]
        turn = (turn + np.pi) % (2 * np.pi) - np.pi
        accel = _safe_divide(np.r_[np.nan, np.diff(speed)], dt)
        angular_velocity = _safe_divide(turn, dt)
        cumulative = np.nancumsum(np.nan_to_num(step, nan=0.0))
        net = np.hypot(x - x[0], y - y[0])
        persistence = _safe_divide(net, cumulative)
        values = [dt, dx, dy, step, speed, vx, vy, accel, turn, angular_velocity,
                  net, cumulative, net, persistence]
        for col, value in zip(motion_cols, values):
            df.loc[idx, col] = value
        df.loc[idx, "track_age"] = t - t[0]
        df.loc[idx, "observation_number"] = np.arange(1, len(idx) + 1)
        for lag in msd_lags:
            sq = np.full(len(idx), np.nan)
            if len(idx) > lag:
                sq[lag:] = (x[lag:] - x[:-lag]) ** 2 + (y[lag:] - y[:-lag]) ** 2
            df.loc[idx, f"squared_displacement_lag_{int(lag)}_um2"] = sq

        for col in temporal_columns:
            vals = df.loc[idx, col].to_numpy(float)
            delta = np.r_[np.nan, np.diff(vals)]
            df.loc[idx, f"{col}__delta"] = delta
            df.loc[idx, f"{col}__rate"] = _safe_divide(delta, dt)
            df.loc[idx, f"{col}__fractional_change"] = _safe_divide(delta, np.r_[np.nan, vals[:-1]])
            df.loc[idx, f"{col}__change_from_start"] = vals - vals[0]
            df.loc[idx, f"{col}__fold_from_start"] = _safe_divide(vals, vals[0])
            df.loc[idx, f"{col}__expanding_mean"] = pd.Series(vals).expanding().mean().to_numpy()
            df.loc[idx, f"{col}__expanding_std"] = pd.Series(vals).expanding(min_periods=2).std(ddof=0).to_numpy()
            df.loc[idx, f"{col}__expanding_min"] = pd.Series(vals).expanding().min().to_numpy()
            df.loc[idx, f"{col}__expanding_max"] = pd.Series(vals).expanding().max().to_numpy()
            df.loc[idx, f"{col}__expanding_slope"] = [_safe_slope(t[:i + 1], vals[:i + 1]) for i in range(len(idx))]
            df.loc[idx, f"{col}__cumulative_abs_change"] = np.nancumsum(np.abs(np.nan_to_num(delta, nan=0.0)))

        for window in rolling_windows:
            tag = str(window).replace(".", "p")
            for i, row_idx in enumerate(idx):
                mask = (t >= t[i] - float(window)) & (t <= t[i])
                # Motility in the window: steps ending at observations in the mask.
                df.at[row_idx, f"speed__rolling_{tag}_mean"] = np.nanmean(speed[mask]) if np.isfinite(speed[mask]).any() else np.nan
                df.at[row_idx, f"speed__rolling_{tag}_std"] = np.nanstd(speed[mask]) if np.isfinite(speed[mask]).any() else np.nan
                df.at[row_idx, f"speed__rolling_{tag}_max"] = np.nanmax(speed[mask]) if np.isfinite(speed[mask]).any() else np.nan
                df.at[row_idx, f"path_length__rolling_{tag}"] = np.nansum(step[mask])
                first = np.flatnonzero(mask)[0]
                wnet = np.hypot(x[i] - x[first], y[i] - y[first])
                wpath = np.nansum(step[mask])
                df.at[row_idx, f"net_displacement__rolling_{tag}_um"] = wnet
                df.at[row_idx, f"persistence__rolling_{tag}"] = wnet / wpath if wpath > 0 else np.nan
                valid_speed = speed[mask & np.isfinite(speed)]
                df.at[row_idx, f"stationary_fraction__rolling_{tag}"] = np.mean(valid_speed <= stationary_speed_threshold) if valid_speed.size else np.nan
                valid_turn = turn[mask & np.isfinite(turn)]
                df.at[row_idx, f"turning_variability__rolling_{tag}"] = np.std(valid_turn) if valid_turn.size else np.nan
                for col in temporal_columns:
                    vals = df.loc[idx, col].to_numpy(float)
                    wvals = vals[mask]
                    prefix = f"{col}__rolling_{tag}"
                    df.at[row_idx, f"{prefix}_mean"] = np.nanmean(wvals) if np.isfinite(wvals).any() else np.nan
                    df.at[row_idx, f"{prefix}_std"] = np.nanstd(wvals) if np.isfinite(wvals).any() else np.nan
                    df.at[row_idx, f"{prefix}_min"] = np.nanmin(wvals) if np.isfinite(wvals).any() else np.nan
                    df.at[row_idx, f"{prefix}_max"] = np.nanmax(wvals) if np.isfinite(wvals).any() else np.nan
                    df.at[row_idx, f"{prefix}_slope"] = _safe_slope(t[mask], wvals)

    df["stationary"] = df["speed_um_per_time"] <= float(stationary_speed_threshold)
    return df.sort_values(["frame_id", "label"]).reset_index(drop=True)


def build_track_summary(observations, temporal_columns=None, msd_lags=(1, 2, 3, 5)):
    """Build one summary row per track."""
    temporal_columns = _resolve_temporal_columns(observations, temporal_columns)
    rows = []
    for tid, g in observations.groupby("track_id", sort=True):
        g = g.sort_values("elapsed_time")
        t = g["elapsed_time"].to_numpy(float)
        x, y = g["centroid_x_um"].to_numpy(float), g["centroid_y_um"].to_numpy(float)
        row = {
            "track_id": int(tid), "parent_track_id": g["parent_track_id"].iloc[0],
            "lineage_id": int(g["lineage_id"].iloc[0]), "generation": int(g["generation"].iloc[0]),
            "start_frame": int(g["frame_id"].min()), "end_frame": int(g["frame_id"].max()),
            "start_time": float(t[0]), "end_time": float(t[-1]), "lifetime": float(t[-1] - t[0]),
            "n_observations": len(g), "n_missing_frames": int(g["gap_frames"].sum()),
            "total_path_length_um": float(g["step_distance_um"].sum()),
            "net_displacement_um": float(np.hypot(x[-1] - x[0], y[-1] - y[0])),
            "mean_speed_um_per_time": float(g["speed_um_per_time"].mean()),
            "max_speed_um_per_time": float(g["speed_um_per_time"].max()),
            "stationary_fraction": float(g["stationary"].mean()),
            "mean_tracking_confidence": float(g["tracking_confidence"].mean()),
            "radius_of_gyration_um": float(np.sqrt(np.mean((x - x.mean()) ** 2 + (y - y.mean()) ** 2))),
        }
        row["directional_persistence"] = row["net_displacement_um"] / row["total_path_length_um"] if row["total_path_length_um"] > 0 else np.nan
        for lag in msd_lags:
            col = f"squared_displacement_lag_{int(lag)}_um2"
            row[f"mean_squared_displacement_lag_{int(lag)}_um2"] = float(g[col].mean()) if col in g else np.nan
        for col in temporal_columns:
            vals = g[col].to_numpy(float)
            row[f"{col}__start"] = vals[0]
            row[f"{col}__end"] = vals[-1]
            row[f"{col}__mean"] = np.nanmean(vals) if np.isfinite(vals).any() else np.nan
            row[f"{col}__std"] = np.nanstd(vals) if np.isfinite(vals).any() else np.nan
            row[f"{col}__min"] = np.nanmin(vals) if np.isfinite(vals).any() else np.nan
            row[f"{col}__max"] = np.nanmax(vals) if np.isfinite(vals).any() else np.nan
            row[f"{col}__net_change"] = vals[-1] - vals[0]
            row[f"{col}__slope"] = _safe_slope(t, vals)
            good = np.isfinite(t) & np.isfinite(vals)
            row[f"{col}__auc"] = float(np.trapz(vals[good], t[good])) if good.sum() >= 2 else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def build_event_table(observations, division_relations, temporal_columns=None):
    """Build birth, end and parent-daughter division relationship events."""
    temporal_columns = _resolve_temporal_columns(observations, temporal_columns)
    events = []
    divided_parents = set(division_relations.get("parent_track_id", [])) if not division_relations.empty else set()
    for tid, g in observations.groupby("track_id", sort=True):
        g = g.sort_values("elapsed_time")
        common = {"track_id": int(tid), "lineage_id": int(g["lineage_id"].iloc[0]), "generation": int(g["generation"].iloc[0])}
        events.append({**common, "event_type": "birth", "frame_id": int(g["frame_id"].iloc[0]), "elapsed_time": float(g["elapsed_time"].iloc[0]), "related_track_id": pd.NA, "event_confidence": np.nan})
        events.append({**common, "event_type": "division" if int(tid) in divided_parents else "track_end", "frame_id": int(g["frame_id"].iloc[-1]), "elapsed_time": float(g["elapsed_time"].iloc[-1]), "related_track_id": pd.NA, "event_confidence": np.nan})

    for event_no, rel in division_relations.reset_index(drop=True).iterrows():
        parent = observations[observations["track_id"] == rel["parent_track_id"]].sort_values("elapsed_time").iloc[-1]
        child_rows = [observations[observations["track_id"] == cid].sort_values("elapsed_time").iloc[0] for cid in rel["child_track_ids"]]
        division_id = f"division_{event_no + 1:06d}"
        metrics = {
            "division_event_id": division_id,
            "combined_area_ratio": rel["combined_area_ratio"],
            "daughter_area_asymmetry": rel["daughter_area_asymmetry"],
        }
        for col in temporal_columns:
            pv = float(parent[col])
            daughters = np.asarray([float(r[col]) for r in child_rows])
            metrics[f"{col}__daughter_sum_over_parent"] = daughters.sum() / pv if np.isfinite(pv) and pv != 0 else np.nan
            metrics[f"{col}__daughter_asymmetry"] = abs(daughters[0] - daughters[1]) / daughters.sum() if np.isfinite(daughters).all() and daughters.sum() != 0 else np.nan
        for child, child_row in zip(rel["child_track_ids"], child_rows):
            events.append({
                "event_type": "division_relationship", "frame_id": int(rel["frame_id"]),
                "elapsed_time": float(rel["elapsed_time"]), "track_id": int(rel["parent_track_id"]),
                "related_track_id": int(child), "lineage_id": int(child_row["lineage_id"]),
                "generation": int(child_row["generation"]), "event_confidence": float(rel["confidence"]),
                **metrics,
            })
    return pd.DataFrame(events).sort_values(["frame_id", "event_type", "track_id"]).reset_index(drop=True)


def build_temporal_tables(
    frame_tables: Sequence[pd.DataFrame] | pd.DataFrame,
    frame_interval=1.0,
    time_unit="frame",
    timestamps=None,
    temporal_columns=None,
    rolling_windows=(3.0, 6.0, 12.0),
    msd_lags=(1, 2, 3, 5),
    stationary_speed_threshold=1.0,
    tracking_kwargs=None,
    output_dir=None,
):
    """Create linked observation, track-summary, and event/lineage tables."""
    if isinstance(frame_tables, pd.DataFrame):
        combined = frame_tables.copy()
    else:
        combined = pd.concat(list(frame_tables), ignore_index=True, sort=False)
    prepared = _prepare_observations(combined, frame_interval, time_unit, timestamps)
    tracked, divisions = link_cell_tracks(prepared, **(tracking_kwargs or {}))
    observations = add_temporal_features(
        tracked, temporal_columns, rolling_windows, msd_lags, stationary_speed_threshold,
    )
    tracks = build_track_summary(observations, temporal_columns, msd_lags)
    events = build_event_table(observations, divisions, temporal_columns)

    divided = set(divisions["parent_track_id"]) if not divisions.empty else set()
    child_count = divisions.explode("child_track_ids").groupby("parent_track_id").size() if not divisions.empty else pd.Series(dtype=int)
    tracks["divided"] = tracks["track_id"].isin(divided)
    tracks["n_daughters"] = tracks["track_id"].map(child_count).fillna(0).astype(int)
    division_time = divisions.set_index("parent_track_id")["elapsed_time"] if not divisions.empty else pd.Series(dtype=float)
    tracks["division_time"] = tracks["track_id"].map(division_time)

    observations["is_division_parent"] = observations["track_id"].isin(divided) & ~observations["track_id"].duplicated(keep="last")
    observations["is_division_child"] = observations["link_type"].eq("division_child")
    observations["time_since_track_start"] = observations["track_age"]
    observations["time_to_division"] = observations["track_id"].map(division_time) - observations["elapsed_time"]
    observations.loc[~observations["track_id"].isin(divided), "time_to_division"] = np.nan
    parent_division_time = observations["parent_track_id"].map(division_time)
    observations["time_since_parent_division"] = observations["elapsed_time"] - parent_division_time
    observations["motility_requires_tracking"] = False

    if output_dir is not None:
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        observations.to_csv(out / "cell_frame_observations.csv", index=False)
        tracks.to_csv(out / "track_summaries.csv", index=False)
        events.to_csv(out / "lineage_events.csv", index=False)
    return observations, tracks, events

