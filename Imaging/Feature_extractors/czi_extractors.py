import numpy as np

# General loader


from pathlib import Path

import numpy as np
from bioio import BioImage
import matplotlib.pyplot as plt


import os
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.segmentation import find_boundaries
from scipy.ndimage import binary_dilation

import matplotlib.patheffects as pe


import imageio.v3 as iio


def load_czi(path, use_aicspylibczi=False):
    """
    Load a Zeiss .czi file using BioIO.

    Parameters
    ----------
    path : str or Path
        Path to .czi file.
    use_aicspylibczi : bool
        If True, uses aicspylibczi backend through bioio-czi.
        Useful for some tiled/mosaic or complex CZI files.

    Returns
    -------
    img : BioImage
        BioIO image object.
    data : np.ndarray
        Image data as numpy array.
    """
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    img = BioImage(path, use_aicspylibczi=use_aicspylibczi)

    print("File:", path.name)
    print("Dimensions:", img.dims)
    print("Dimension order:", img.dims.order)
    print("Shape:", img.shape)
    print("Scenes:", img.scenes)

    data = img.data

    return img, data

def get_tczyx(img):
    """
    Return image as TCZYX.

    Output shape:
        T, C, Z, Y, X

    Missing dimensions are filled with size 1.
    """
    arr = img.get_image_data("TCZYX")
    return arr

def show_plane(tczyx, t=0, c=0, z=0, savefile='plot', save=True):
    """
    Display one 2D image plane from TCZYX data.
    """
    plane = tczyx[t, c, z]

    plt.figure(figsize=(6, 6))
    plt.imshow(plane, cmap="gray")
    # plt.title(f"T={t}, C={c}, Z={z}")
    plt.axis("off")

    if save:
        plt.savefig(f'{savefile}.png', dpi=300, bbox_inches='tight')
    
    plt.show()

def save_xy_png(tczyx, t=0, c=0, z=0, savefile='plot'):
    """
    save tensor into png directly
    """
    plane = tczyx[t, c, z]

    pure_save_image(plane, savefile)

def pure_save_image(plane, savefile):
    img = np.asarray(plane)

    img = np.nan_to_num(img, nan=0, posinf=0, neginf=0)

    if img.dtype != np.uint16:
        img = img.astype(float)

        img = img - img.min()

        if img.max() > 0:
            img = img / img.max()

        img = (img * 65535).astype(np.uint16)

    iio.imwrite(savefile, img)

def extract_mosaic_tiles(tensor, shp, scene=0, time=0, z=0):
    """
    Extract mosaic tiles for one scene and one timepoint.

    Your tensor axes:
    H, S, T, C, Z, M, Y, X

    Returns
    -------
    tiles : np.ndarray
        Shape = (M, C, Y, X)
    """

    axes = [a for a, n in shp]
    ax = {name: i for i, name in enumerate(axes)}

    # select fixed H, S, T, Z
    index = [slice(None)] * tensor.ndim

    if "H" in ax.keys():
        index[ax["H"]] = 0
    if "S" in ax.keys():
        index[ax["S"]] = scene
    if "Z" in ax.keys():
        index[ax["Z"]] = z
    if "T" in ax.keys():
        index[ax["T"]] = time

    print(f"Extracting with index : {index}")
    
    out = tensor[tuple(index)]

    # after indexing, remaining axes should include C, M, Y, X
    remaining_axes = [a for a in axes if a not in ["H", "S", "T", "Z"]]

    # reorder to M, C, Y, X
    order = [remaining_axes.index(a) for a in ["M", "C", "Y", "X"]]
    tiles = np.transpose(out, order)

    return tiles
import numpy as np
from aicspylibczi import CziFile

def normalize_tile(tile, q_low=1, q_high=99.8):
    tile = tile.astype(np.float32)
    lo = np.percentile(tile, q_low, axis=(-2, -1), keepdims=True)
    hi = np.percentile(tile, q_high, axis=(-2, -1), keepdims=True)
    return (tile - lo) / np.maximum(hi - lo, 1e-6)

def make_soft_weight(h, w, edge_fraction=0.04, min_weight=0.85):
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

def stitch_metadata_clean(
    czi_path,
    tiles,
    scene=0,
    time=0,
    z=0,
    channel_ref=0,
    crop_edge=10,
    normalize=True,
    offset_x=0,
    offset_y=0,
    edge_fraction=0.04,
    min_weight=0.85,
):
    czi = CziFile(czi_path)
    M, C, Y, X = tiles.shape
    dtype = tiles.dtype

    positions = []
    for m in range(M):
        bb = czi.get_mosaic_tile_bounding_box(
            M=m, S=scene, T=time, C=channel_ref, Z=z
        )

        positions.append({
            "M": m,
            "x": int(bb.x),
            "y": int(bb.y),
            "w": int(bb.w),
            "h": int(bb.h),
        })

    # infer rows/cols from metadata positions
    xs = sorted(set(p["x"] for p in positions))
    ys = sorted(set(p["y"] for p in positions))

    for p in positions:
        col = np.argmin([abs(p["x"] - x) for x in xs])
        row = np.argmin([abs(p["y"] - y) for y in ys])

        p["row"] = int(row)
        p["col"] = int(col)

        # apply global spacing correction
        p["x_adj"] = p["x"] + p["col"] * offset_x
        p["y_adj"] = p["y"] + p["row"] * offset_y

    min_x = min(p["x_adj"] for p in positions)
    min_y = min(p["y_adj"] for p in positions)

    for p in positions:
        p["x0"] = int(p["x_adj"] - min_x)
        p["y0"] = int(p["y_adj"] - min_y)

    max_x = max(p["x0"] + X for p in positions)
    max_y = max(p["y0"] + Y for p in positions)

    
    canvas = np.zeros((C, max_y, max_x), dtype=np.float32)
    weights = np.zeros((1, max_y, max_x), dtype=np.float32)
    
    for p in positions:
        m = p["M"]
        tile = tiles[m].astype(np.float32)

        if normalize:
            tile_norm = normalize_tile(tile)
            scale = np.percentile(tile, 99.8, axis=(-2, -1), keepdims=True)
            tile = tile_norm * scale

        ce = crop_edge
        tile = tile[:, ce:Y-ce, ce:X-ce]

        h, w = tile.shape[-2:]

        y = p["y0"] + ce
        x = p["x0"] + ce

    
        weight = make_soft_weight(
            h, w,
            edge_fraction=edge_fraction,
            min_weight=min_weight,
        )[None, :, :]
        
        canvas[:, y:y+h, x:x+w] += tile * weight
        weights[:, y:y+h, x:x+w] += weight

    stitched = canvas / np.maximum(weights, 1e-6)

    if np.issubdtype(dtype, np.integer):
        stitched = np.clip(stitched, 0, np.iinfo(dtype).max)

    return stitched.astype(dtype), positions

def make_old_feather_weight(h, w, edge_fraction=0.12):
    yy = np.ones(h, dtype=np.float32)
    xx = np.ones(w, dtype=np.float32)

    ey = max(1, int(h * edge_fraction))
    ex = max(1, int(w * edge_fraction))

    ramp_y = np.linspace(0, 1, ey, dtype=np.float32)
    ramp_x = np.linspace(0, 1, ex, dtype=np.float32)

    yy[:ey] = ramp_y
    yy[-ey:] = ramp_y[::-1]

    xx[:ex] = ramp_x
    xx[-ex:] = ramp_x[::-1]

    return np.outer(yy, xx)


def stitch_metadata_feathered_old(
    czi_path,
    tiles,
    scene=0,
    time=0,
    z=0,
    channel_ref=0,
    crop_edge=0,
    normalize=True,
    offset_x=-20,
    offset_y=-20,
    edge_fraction=0.12,
):
    czi = CziFile(czi_path)
    M, C, Y, X = tiles.shape
    dtype = tiles.dtype

    positions = []
    for m in range(M):
        bb = czi.get_mosaic_tile_bounding_box(
            M=m, S=scene, T=time, C=channel_ref, Z=z
        )
        positions.append({
            "M": m,
            "x": int(bb.x),
            "y": int(bb.y),
        })

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

    max_x = max(p["x0"] + X for p in positions)
    max_y = max(p["y0"] + Y for p in positions)

    canvas = np.zeros((C, max_y, max_x), dtype=np.float32)
    weights = np.zeros((1, max_y, max_x), dtype=np.float32)

    for p in positions:
        m = p["M"]
        tile = tiles[m].astype(np.float32)

        if normalize:
            tile_norm = normalize_tile(tile)
            scale = np.percentile(tile, 99.8, axis=(-2, -1), keepdims=True)
            tile = tile_norm * scale

        ce = crop_edge
        tile = tile[:, ce:Y-ce if ce else Y, ce:X-ce if ce else X]

        h, w = tile.shape[-2:]

        weight = make_old_feather_weight(
            h,
            w,
            edge_fraction=edge_fraction,
        )[None, :, :]

        y = p["y0"] + ce
        x = p["x0"] + ce

        canvas[:, y:y+h, x:x+w] += tile * weight
        weights[:, y:y+h, x:x+w] += weight

    stitched = canvas / np.maximum(weights, 1e-6)

    if np.issubdtype(dtype, np.integer):
        stitched = np.clip(stitched, 0, np.iinfo(dtype).max)

    return stitched.astype(dtype), positions



def normalize_for_display(img, p1=1, p99=99.8):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, [p1, p99])
    return np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1)


def make_red_mask_overlay(
    raw_img,
    labels,
    mask_alpha=0.35,
    boundary_color=(1, 1, 1),
    boundary_width=2,
    darken_background=0.65,
):
    """
    raw_img: 2D grayscale image
    labels: 2D label mask
    """

    img = normalize_for_display(raw_img)
    base = np.dstack([img, img, img]) * darken_background

    mask = labels > 0
    boundaries = find_boundaries(labels, mode="outer")

    if boundary_width > 1:
        boundaries = binary_dilation(boundaries, iterations=boundary_width)

    overlay = base.copy()

    # red mask fill
    red = np.zeros_like(base)
    red[..., 0] = 1.0

    overlay[mask] = (
        (1 - mask_alpha) * overlay[mask]
        + mask_alpha * red[mask]
    )

    # boundary color
    overlay[boundaries] = boundary_color

    return overlay

def save_segmentation_panels(
    image_cyx3,
    labels,
    props,
    seg_channel_idx,
    out_dir="segmentation_panels",
    prefix="scene0_time000",
    mask_alpha=0.35,
):
    os.makedirs(out_dir, exist_ok=True)

    raw_seg = rgb2gray(image_cyx3[seg_channel_idx])
    raw_disp = normalize_for_display(raw_seg)

    overlay = make_red_mask_overlay(
        raw_seg,
        labels,
        mask_alpha=mask_alpha,
        boundary_color=(1, 1, 1),   # white boundaries
        boundary_width=2,
        darken_background=0.60,
    )

    # 1. Raw segmentation channel
    plt.figure(figsize=(8, 8))
    plt.imshow(raw_disp, cmap="gray")
    plt.title("Segmentation channel")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_seg_channel.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 2. Label mask
    plt.figure(figsize=(8, 8))
    plt.imshow(labels, cmap="nipy_spectral")
    plt.title("Label mask")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_label_mask.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 3. Red mask overlay
    plt.figure(figsize=(8, 8))
    plt.imshow(overlay)
    plt.title("Segmentation mask overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(f"{out_dir}/{prefix}_red_mask_overlay.png", dpi=300, bbox_inches="tight")
    plt.show()

    # 4. Overlay with label IDs
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(overlay)

    
    text_dx = 8
    text_dy = -8
    
    for _, row in props.iterrows():
        ax.text(
            row["centroid_x_px"] + text_dx,
            row["centroid_y_px"] + text_dy,
            str(int(row["label"])),
            color="yellow",
            fontsize=7,
            ha="left",
            va="bottom",
            path_effects=[
                pe.withStroke(linewidth=2, foreground="black")
            ],
        )

    ax.set_title("Segmentation mask overlay with labels")
    ax.axis("off")
    plt.tight_layout()
    fig.savefig(f"{out_dir}/{prefix}_red_mask_overlay_labels.png", dpi=300, bbox_inches="tight")
    plt.show()
