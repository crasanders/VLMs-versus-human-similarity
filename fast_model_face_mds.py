"""Fit a 4D nonmetric MDS solution to Jev face similarity ratings and rotate it onto the physical dimensions.

Follows Nosofsky (1991, p. 5) and fast_model_mds.py:
- Dissimilarity = 10 - similarity rating (matching Nosofsky 1991's 1-10 scale)
- 4D nonmetric MDS
- Orthogonal Procrustes rotation onto the z-scored physical dimensions (EH, ES, NL, MH) in
  data/nosofsky1991_table_a1.csv (oriented with Table A1's sign convention: -EH, +ES, +NL, -MH),
  uniformly scaled to match the Frobenius norm of the centered human MDS solution.

Usage:
    python fast_model_face_mds.py [--similarities CSV] [--output-folder DIR]
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist
from scipy.ndimage import binary_fill_holes
from sklearn.manifold import MDS

from make_jev_plots import GRID, HUMAN_COLOR, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, style_axes

HUMAN_DATA_FILE = Path("data") / "nosofsky1991_table_a1.csv"
FACE_IMAGE_FOLDER = Path("data") / "34 Faces"
DEFAULT_SIMILARITIES = Path("output") / "jev" / "face_similarities.csv"
DEFAULT_OUTPUT_FOLDER = Path("output") / "face_mds"
DUPLICATE_FACES = {26, 30}  # Identical to Faces 11 and 18 in Nosofsky (1991)

N_DIMENSIONS = 4
RANDOM_STATE = 0
JEV_COLOR = "#2a78d6"
CAT1_EXEMPLARS = {1, 2, 3, 4, 5}
CAT2_EXEMPLARS = {6, 7, 8, 9, 10}
CAT1_COLOR = "#184f95"
CAT2_COLOR = "#a83232"

DIMENSIONS = [
    ("EH", "MDS_1", "Dim 1 (Eye Height)"),
    ("ES", "MDS_2", "Dim 2 (Eye Separation)"),
    ("NL", "MDS_3", "Dim 3 (Nose Length)"),
    ("MH", "MDS_4", "Dim 4 (Mouth Height)"),
]
DIM_LABELS = [label for _, _, label in DIMENSIONS]

DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#a83232", "#f0efec", "#184f95"])


def load_human_data():
    human = pd.read_csv(HUMAN_DATA_FILE)
    human = human[~human["Face"].isin(DUPLICATE_FACES)].copy()
    human["Face Name"] = [f"face_{int(f):02d}" for f in human["Face"]]
    human = human.set_index("Face Name")
    return human


def load_dissimilarities(path: Path, faces: list[str]) -> np.ndarray:
    data = pd.read_csv(path)
    dissims = np.zeros((len(faces), len(faces)))
    filled = np.eye(len(faces), dtype=bool)
    for _, row in data.iterrows():
        i, j = faces.index(row["Face1"]), faces.index(row["Face2"])
        rating = float(row["Model Rating"]) if "Model Rating" in row else float(row["ChatGPT Rating"])
        dissims[i, j] = dissims[j, i] = max(1e-6, 10.0 - rating)
        filled[i, j] = filled[j, i] = True
    if not filled.all():
        raise ValueError(f"{path} is missing {(~filled).sum() // 2} pairs")
    return dissims


def fit_mds(dissims: np.ndarray) -> tuple[np.ndarray, float]:
    mds = MDS(
        n_components=N_DIMENSIONS,
        metric_mds=False,
        n_init=100,
        max_iter=3000,
        eps=1e-6,
        init="random",
        metric="precomputed",
        normalized_stress=True,
        random_state=RANDOM_STATE,
        n_jobs=1,
    )
    return mds.fit_transform(dissims), float(mds.stress_)


# Sign convention matching Nosofsky (1991) Table A1 MDS axes (-EH, +ES, +NL, -MH)
PHYS_SIGNS = np.array([-1.0, 1.0, 1.0, -1.0])


def rotate_onto_physical(
    solution: np.ndarray,
    physical_dims: np.ndarray,
    human_mds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal Procrustes rotation onto z-scored physical dimensions (-EH, +ES, +NL, -MH),
    uniformly scaled to match the Frobenius norm of the human MDS solution.
    Returns (rotated_jev_mds, scaled_physical_target).
    """
    physical_dims = np.asarray(physical_dims, dtype=float)
    human_mds = np.asarray(human_mds, dtype=float)

    # Orient signs to match Table A1 MDS convention and z-score each physical dimension
    phys_signed = physical_dims * PHYS_SIGNS[None, :]
    phys_z = (phys_signed - phys_signed.mean(axis=0)) / phys_signed.std(axis=0, ddof=0)
    phys_norm = np.linalg.norm(phys_z)

    human_mean = human_mds.mean(axis=0)
    human_centered = human_mds - human_mean
    human_norm = np.linalg.norm(human_centered)

    centered = solution - solution.mean(axis=0)
    sol_norm = np.linalg.norm(centered)
    rotation, _ = orthogonal_procrustes(phys_z / phys_norm, centered / sol_norm)
    rotated_unit = (centered / sol_norm) @ rotation.T
    rotated = rotated_unit * human_norm + human_mean

    # Also express the z-scored physical target on the same Frobenius norm for plotting
    phys_scaled = (phys_z / phys_norm) * human_norm + human_mean

    # Verify relative pairwise distances are preserved up to the uniform scale factor
    scale_factor = human_norm / sol_norm
    assert np.allclose(pdist(solution) * scale_factor, pdist(rotated)), "rotation distorted relative pairwise distances"
    return rotated, phys_scaled


def correlate_matrices(a: np.ndarray, b: np.ndarray, row_labels: list[str], col_labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        [[np.corrcoef(a[:, i], b[:, j])[0, 1] for j in range(b.shape[1])] for i in range(a.shape[1])],
        index=row_labels,
        columns=col_labels,
    )


def save(fig: plt.Figure, path: Path) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(table: pd.DataFrame, title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1.3 * table.shape[1] + 2.8, 0.55 * table.shape[0] + 1.5))
    arr = table.to_numpy(dtype=float)
    ax.imshow(arr, cmap=DIVERGING, vmin=-1, vmax=1, aspect="auto")
    for (i, j), value in np.ndenumerate(arr):
        if np.isnan(value):
            continue
        ax.text(
            j,
            i,
            f"{value:.2f}",
            ha="center",
            va="center",
            fontsize=9,
            color=SURFACE if abs(value) >= 0.6 else TEXT_PRIMARY,
        )
    ax.set_xticks(range(table.shape[1]), table.columns, rotation=25, ha="right")
    ax.set_yticks(range(table.shape[0]), table.index)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left")
    fig.tight_layout()
    save(fig, path)


def place_non_overlapping_labels(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    colors: list[str],
    weights: list[str],
    lim: tuple[float, float],
    fontsize: float = 7.0,
    point_radius_norm: float = 0.022,
    box_w_norm: float = 0.046,
    box_h_norm: float = 0.036,
    zorder: int = 10,
) -> None:
    """Place text labels around (x, y) points with zero label-box overlap and leader lines when displaced."""
    n = len(x)
    span = float(lim[1] - lim[0])
    u = (np.asarray(x, dtype=float) - lim[0]) / span
    v = (np.asarray(y, dtype=float) - lim[0]) / span

    # Per-label half-widths and half-heights in normalized [0, 1] axis units
    hw = np.array([box_w_norm * (0.42 if len(str(lbl)) == 1 else 0.54) for lbl in labels], dtype=float)
    hh = np.full(n, box_h_norm * 0.52, dtype=float)

    # Compute local cluster offset direction for each point to break ties cleanly
    lu = np.zeros(n, dtype=float)
    lv = np.zeros(n, dtype=float)
    placed = np.zeros(n, dtype=bool)

    # Order points from most crowded to least crowded so dense clusters get first pick of close ring slots
    dist_mat = np.hypot(u[:, None] - u[None, :], v[:, None] - v[None, :])
    crowding = np.sum(np.exp(-((dist_mat / 0.08) ** 2)), axis=1)
    order = np.argsort(-crowding)

    radii = np.linspace(point_radius_norm + 0.012, 0.28, 16)
    n_angles = 36
    base_angles = np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)

    margin = 0.025
    for idx_rank, i in enumerate(order):
        # Local neighbor centroid to bias outward direction
        neighbors = dist_mat[i] < 0.12
        if np.sum(neighbors) > 1:
            cu = np.mean(u[neighbors])
            cv = np.mean(v[neighbors])
            out_angle = np.arctan2(v[i] - cv, u[i] - cu)
            if np.hypot(u[i] - cu, v[i] - cv) < 1e-4:
                out_angle = (2.0 * np.pi * idx_rank) / n
        else:
            out_angle = np.pi / 4.0

        best_cost = np.inf
        best_pos = (np.clip(u[i] + 0.025, margin, 1.0 - margin), np.clip(v[i] + 0.025, margin, 1.0 - margin))

        for r in radii:
            angles = out_angle + base_angles
            for ang in angles:
                cu_cand = u[i] + r * np.cos(ang)
                cv_cand = v[i] + r * np.sin(ang)
                if (
                    cu_cand - hw[i] < margin
                    or cu_cand + hw[i] > 1.0 - margin
                    or cv_cand - hh[i] < margin
                    or cv_cand + hh[i] > 1.0 - margin
                ):
                    continue

                # Check overlap with already-placed label boxes
                if np.any(placed):
                    dx_box = np.abs(cu_cand - lu[placed]) - (hw[i] + hw[placed] + 0.003)
                    dy_box = np.abs(cv_cand - lv[placed]) - (hh[i] + hh[placed] + 0.003)
                    if np.any((dx_box < 0.0) & (dy_box < 0.0)):
                        continue

                # Check clearance from all data point markers
                d_pts = np.hypot(cu_cand - u, cv_cand - v)
                min_pt_clear = point_radius_norm + 0.75 * min(hw[i], hh[i])
                pt_overlap = np.maximum(0.0, min_pt_clear - d_pts)
                pt_penalty = 80.0 * float(np.sum(pt_overlap ** 2))

                # Distance penalty + slight angular preference for outward direction
                ang_diff = np.abs(np.arctan2(np.sin(ang - out_angle), np.cos(ang - out_angle)))
                cost = r + 0.008 * ang_diff + pt_penalty
                if cost < best_cost:
                    best_cost = cost
                    best_pos = (cu_cand, cv_cand)
            if best_cost < 0.45:
                break

        lu[i], lv[i] = best_pos
        placed[i] = True

    # Iterative relaxation to guarantee zero label-label overlap and good point separation
    for it in range(200):
        # 1. Label-label box repulsion
        for i in range(n):
            for j in range(i + 1, n):
                ox = (hw[i] + hw[j] + 0.005) - abs(lu[i] - lu[j])
                oy = (hh[i] + hh[j] + 0.005) - abs(lv[i] - lv[j])
                if ox > 0.0 and oy > 0.0:
                    if ox < oy:
                        sgn = 1.0 if lu[i] >= lu[j] else -1.0
                        shift = 0.52 * ox * sgn
                        lu[i] += shift
                        lu[j] -= shift
                    else:
                        sgn = 1.0 if lv[i] >= lv[j] else -1.0
                        shift = 0.52 * oy * sgn
                        lv[i] += shift
                        lv[j] -= shift

        # 2. Point-marker repulsion and weak tether during first 130 iterations
        if it < 130:
            for i in range(n):
                for j in range(n):
                    dx = lu[i] - u[j]
                    dy = lv[i] - v[j]
                    d = np.hypot(dx, dy)
                    min_d = point_radius_norm + 0.65 * min(hw[i], hh[i])
                    if d < min_d:
                        if d < 1e-5:
                            ang = (2.0 * np.pi * i) / n
                            dx, dy, d = np.cos(ang), np.sin(ang), 1.0
                        push = 0.38 * (min_d - d)
                        lu[i] += push * (dx / d)
                        lv[i] += push * (dy / d)

                # Gentle pull toward anchor if far
                dx_a = u[i] - lu[i]
                dy_a = v[i] - lv[i]
                da = np.hypot(dx_a, dy_a)
                if da > point_radius_norm + 0.035:
                    lu[i] += 0.025 * dx_a
                    lv[i] += 0.025 * dy_a

        for i in range(n):
            lu[i] = np.clip(lu[i], margin + hw[i], 1.0 - margin - hw[i])
            lv[i] = np.clip(lv[i], margin + hh[i], 1.0 - margin - hh[i])

    # Render annotations with leader lines when displaced
    lx = lim[0] + lu * span
    ly = lim[0] + lv * span
    for i in range(n):
        disp_norm = np.hypot(lu[i] - u[i], lv[i] - v[i])
        use_arrow = disp_norm > (point_radius_norm + 0.015)
        arrowprops = (
            {
                "arrowstyle": "-",
                "color": colors[i],
                "lw": 0.65,
                "alpha": 0.6,
                "shrinkA": 0,
                "shrinkB": 2,
            }
            if use_arrow
            else None
        )
        ax.annotate(
            str(labels[i]),
            xy=(x[i], y[i]),
            xytext=(lx[i], ly[i]),
            textcoords="data",
            ha="center",
            va="center",
            fontsize=fontsize,
            fontweight=weights[i],
            color=colors[i],
            bbox={"boxstyle": "round,pad=0.13", "fc": SURFACE, "ec": GRID, "lw": 0.4, "alpha": 0.96},
            arrowprops=arrowprops,
            zorder=zorder,
        )


def plot_dimension_scatter(
    target_coords: np.ndarray,
    jev_mds: np.ndarray,
    face_numbers: list[int],
    path: Path,
    xlabel: str = "Human MDS coordinate",
    ylabel: str = "Jev rotated MDS coordinate",
    suptitle: str = "Jev (physical-rotated) vs. human 4D MDS coordinates (32 unique schematic faces, Nosofsky 1991)",
) -> None:
    """4-panel scatterplot comparing Target (Human MDS or Physical) vs. Model Rotated MDS across all 4 dimensions."""
    fig, axes = plt.subplots(1, N_DIMENSIONS, figsize=(4.0 * N_DIMENSIONS, 4.2), sharex=True, sharey=True)
    lim = (-1.5, 1.5)
    ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]
    labels = [str(f) for f in face_numbers]
    colors = [
        CAT1_COLOR if f in CAT1_EXEMPLARS else (CAT2_COLOR if f in CAT2_EXEMPLARS else TEXT_PRIMARY)
        for f in face_numbers
    ]
    weights = ["bold" if f in (CAT1_EXEMPLARS | CAT2_EXEMPLARS) else "normal" for f in face_numbers]

    for d, ax in enumerate(axes):
        style_axes(ax)
        hx = target_coords[:, d]
        jy = jev_mds[:, d]
        r = np.corrcoef(hx, jy)[0, 1]

        ax.plot(lim, lim, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
        ax.scatter(hx, jy, s=28, color=JEV_COLOR, edgecolor=SURFACE, linewidth=0.8, zorder=2)

        place_non_overlapping_labels(
            ax,
            hx,
            jy,
            labels=labels,
            colors=colors,
            weights=weights,
            lim=lim,
            fontsize=6.8,
            point_radius_norm=0.018,
            box_w_norm=0.052,
            box_h_norm=0.040,
            zorder=6,
        )

        ax.set_title(f"{DIM_LABELS[d]}\nr = {r:.2f}", loc="left", fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        if d == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(
        suptitle,
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


_thumbnails: dict[str, np.ndarray] = {}


def load_face_thumbnail(face: str) -> np.ndarray:
    """Load a face PNG and make the exterior outside the head outline transparent with bold lines for small thumbnails."""
    if face not in _thumbnails:
        im = Image.open(FACE_IMAGE_FOLDER / f"{face}.png").convert("RGBA")
        arr = np.array(im)
        # Head outline is dark pixels; fill inside of head outline so interior stays opaque white
        is_ink = arr[:, :, 0] < 240
        inside_head = binary_fill_holes(is_ink)
        arr[~inside_head, 3] = 0
        # Crop tightly around the head outline
        ys, xs = np.where(inside_head)
        arr = arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
        # Downsample with LANCZOS and darken ink strokes so features remain crisp at thumbnail size
        thumb_h = 48
        thumb_w = int(round(arr.shape[1] * (thumb_h / arr.shape[0])))
        thumb = np.array(Image.fromarray(arr).resize((thumb_w, thumb_h), Image.Resampling.LANCZOS), dtype=float) / 255.0
        # Darken gray anti-aliased strokes (gamma > 1 on luminance) while keeping white interior white
        thumb[:, :, :3] = np.clip((thumb[:, :, :3] ** 2.4), 0.0, 1.0)
        _thumbnails[face] = thumb
    return _thumbnails[face]


def plot_dimension_scatter_faces(
    target_coords: np.ndarray,
    jev_mds: np.ndarray,
    faces: list[str],
    face_numbers: list[int],
    path: Path,
    xlabel: str = "Human MDS coordinate",
    ylabel: str = "Jev rotated MDS coordinate",
    suptitle: str = "Jev (physical-rotated) vs. human 4D MDS coordinates (32 unique schematic faces, Nosofsky 1991)",
) -> None:
    """4-panel scatterplot comparing Target (Human MDS or Physical) vs. Model Rotated MDS using the faces as points."""
    fig, axes = plt.subplots(1, N_DIMENSIONS, figsize=(4.2 * N_DIMENSIONS, 4.4), sharex=True, sharey=True)
    lim = (-1.5, 1.5)
    ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    for d, ax in enumerate(axes):
        style_axes(ax)
        hx = target_coords[:, d]
        jy = jev_mds[:, d]
        r = np.corrcoef(hx, jy)[0, 1]

        ax.plot(lim, lim, color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
        for face, f_num, x0, y0 in zip(faces, face_numbers, hx, jy):
            image = load_face_thumbnail(face)
            z_base = 4 if f_num in (CAT1_EXEMPLARS | CAT2_EXEMPLARS) else 2
            ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.42), (x0, y0), frameon=False, zorder=z_base))
            if f_num in CAT1_EXEMPLARS:
                ax.scatter([x0], [y0], marker="o", s=700, facecolors="none", edgecolors=CAT1_COLOR, linewidths=1.5, zorder=z_base + 1)
            elif f_num in CAT2_EXEMPLARS:
                ax.scatter([x0], [y0 - 0.01], marker="^", s=900, facecolors="none", edgecolors=CAT2_COLOR, linewidths=1.5, zorder=z_base + 1)

        labels = [str(f) for f in face_numbers]
        colors = [
            CAT1_COLOR if f in CAT1_EXEMPLARS else (CAT2_COLOR if f in CAT2_EXEMPLARS else TEXT_PRIMARY)
            for f in face_numbers
        ]
        weights = ["bold" if f in (CAT1_EXEMPLARS | CAT2_EXEMPLARS) else "normal" for f in face_numbers]
        place_non_overlapping_labels(
            ax,
            hx,
            jy,
            labels=labels,
            colors=colors,
            weights=weights,
            lim=lim,
            fontsize=6.6,
            point_radius_norm=0.042,
            box_w_norm=0.052,
            box_h_norm=0.040,
            zorder=10,
        )

        ax.set_title(f"{DIM_LABELS[d]}\nr = {r:.2f}", loc="left", fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        if d == 0:
            ax.set_ylabel(ylabel)

    fig.suptitle(
        suptitle,
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, path)


def plot_mds_space(
    solution: np.ndarray,
    faces: list[str],
    face_numbers: list[int],
    title: str,
    path: Path,
) -> None:
    """2-panel 2D MDS space plot (Dim 1 vs. 2 and Dim 3 vs. 4) with face thumbnails, number labels, and exemplar shapes."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.5))
    lim = (-1.5, 1.5)
    ticks = [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]

    for ax_idx, (ax, (d1, d2)) in enumerate(zip(axes, [(0, 1), (2, 3)])):
        style_axes(ax)
        x, y = solution[:, d1], solution[:, d2]

        for face, f_num, x0, y0 in zip(faces, face_numbers, x, y):
            image = load_face_thumbnail(face)
            z_base = 4 if f_num in (CAT1_EXEMPLARS | CAT2_EXEMPLARS) else 2
            ab = AnnotationBbox(OffsetImage(image, zoom=0.55), (x0, y0), frameon=False, zorder=z_base)
            ax.add_artist(ab)

            # Enclose Category 1 (1-5) in circles and Category 2 (6-10) in triangles
            if f_num in CAT1_EXEMPLARS:
                ax.scatter(
                    [x0],
                    [y0],
                    marker="o",
                    s=1050,
                    facecolors="none",
                    edgecolors=CAT1_COLOR,
                    linewidths=1.8,
                    zorder=z_base + 1,
                )
            elif f_num in CAT2_EXEMPLARS:
                ax.scatter(
                    [x0],
                    [y0 - 0.015],
                    marker="^",
                    s=1350,
                    facecolors="none",
                    edgecolors=CAT2_COLOR,
                    linewidths=1.8,
                    zorder=z_base + 1,
                )

        labels = [str(f) for f in face_numbers]
        colors = [
            CAT1_COLOR if f in CAT1_EXEMPLARS else (CAT2_COLOR if f in CAT2_EXEMPLARS else TEXT_PRIMARY)
            for f in face_numbers
        ]
        weights = ["bold" if f in (CAT1_EXEMPLARS | CAT2_EXEMPLARS) else "normal" for f in face_numbers]
        place_non_overlapping_labels(
            ax,
            x,
            y,
            labels=labels,
            colors=colors,
            weights=weights,
            lim=lim,
            fontsize=7.4,
            point_radius_norm=0.036,
            box_w_norm=0.040,
            box_h_norm=0.032,
            zorder=8,
        )

        if ax_idx == 0:
            ax.scatter([], [], marker="o", s=85, facecolors="none", edgecolors=CAT1_COLOR, linewidths=1.8, label="Category 1 training (1–5)")
            ax.scatter([], [], marker="^", s=90, facecolors="none", edgecolors=CAT2_COLOR, linewidths=1.8, label="Category 2 training (6–10)")
            ax.legend(loc="lower left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8.5)

        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlabel(DIM_LABELS[d1])
        ax.set_ylabel(DIM_LABELS[d2])

    fig.suptitle(title, x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--similarities",
        type=Path,
        default=DEFAULT_SIMILARITIES,
        help=f"Pairwise face similarity ratings CSV (default: {DEFAULT_SIMILARITIES})",
    )
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=DEFAULT_OUTPUT_FOLDER,
        help=f"Output folder for MDS CSVs and plots (default: {DEFAULT_OUTPUT_FOLDER})",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model display name for plot labels and filenames (e.g., 'Jev' or 'GPT-4o')",
    )
    args = parser.parse_args()
    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    if args.model_name:
        model_name = args.model_name
    elif "gpt4o" in str(args.similarities).lower() or "gpt4o" in str(output_folder).lower():
        model_name = "GPT-4o"
    else:
        model_name = "Jev"
    model_slug = model_name.lower().replace("-", "").replace(" ", "_")

    human = load_human_data()
    faces = list(human.index)
    face_numbers = [int(f) for f in human["Face"]]
    human_mds = human[[mds_col for _, mds_col, _ in DIMENSIONS]].to_numpy(dtype=float)
    physical_dims = human[[phys_col for phys_col, _, _ in DIMENSIONS]].to_numpy(dtype=float)

    print(f"Fitting 4D nonmetric MDS to {args.similarities} ({model_name})...")
    dissims = load_dissimilarities(args.similarities, faces)
    solution, stress = fit_mds(dissims)
    rotated, phys_scaled = rotate_onto_physical(solution, physical_dims, human_mds)

    # Save rotated MDS coordinates alongside human MDS and scaled physical coordinates
    col_prefix = "Jev" if model_slug == "jev" else "GPT4o"
    mds_df = pd.DataFrame(
        rotated,
        index=faces,
        columns=[f"{col_prefix}_MDS_{d + 1}" for d in range(N_DIMENSIONS)],
    )
    mds_df.insert(0, "Face", face_numbers)
    if col_prefix != "Jev":
        for d in range(N_DIMENSIONS):
            mds_df[f"Jev_MDS_{d + 1}"] = rotated[:, d]
    for d in range(N_DIMENSIONS):
        mds_df[f"Human_MDS_{d + 1}"] = human_mds[:, d]
    for d, (phys_col, _, _) in enumerate(DIMENSIONS):
        mds_df[f"Physical_{phys_col}"] = physical_dims[:, d]
        mds_df[f"Physical_Scaled_{d + 1}"] = phys_scaled[:, d]
    mds_df.to_csv(output_folder / f"{model_slug}_mds.csv", index_label="Face Name")
    if model_slug != "jev":
        mds_df.to_csv(output_folder / "jev_mds.csv", index_label="Face Name")

    # Correlation matrices: Model rotated MDS (rows) vs. Human MDS (columns) and Physical dimensions (columns)
    vs_human = correlate_matrices(rotated, human_mds, [f"{model_name} {l}" for l in DIM_LABELS], [f"Human {l}" for l in DIM_LABELS])
    vs_phys_raw = correlate_matrices(rotated, physical_dims, [f"{model_name} {l}" for l in DIM_LABELS], [f"Physical {p}" for p, _, _ in DIMENSIONS])
    vs_phys_oriented = correlate_matrices(
        rotated,
        phys_scaled,
        [f"{model_name} {l}" for l in DIM_LABELS],
        [f"Physical {'-EH' if p == 'EH' else ('-MH' if p == 'MH' else '+' + p)}" for p, _, _ in DIMENSIONS],
    )
    vs_human.round(3).to_csv(output_folder / f"{model_slug}_vs_human_mds_correlations.csv")
    vs_phys_raw.round(3).to_csv(output_folder / f"{model_slug}_vs_physical_correlations.csv")

    ylabel = f"{model_name} rotated MDS coordinate"
    plot_heatmap(
        vs_human,
        f"{model_name} (physical-rotated) MDS vs. human MDS dimensions (stress = {stress:.3f})",
        output_folder / f"{model_slug}_vs_human_mds_correlations",
    )
    plot_heatmap(
        vs_phys_oriented,
        f"{model_name} (physical-rotated) MDS vs. physical dimensions (stress = {stress:.3f})",
        output_folder / f"{model_slug}_vs_physical_correlations",
    )
    plot_dimension_scatter(
        human_mds,
        rotated,
        face_numbers,
        output_folder / f"{model_slug}_vs_human_mds_scatter",
        xlabel="Human MDS coordinate",
        ylabel=ylabel,
        suptitle=f"{model_name} (physical-rotated) vs. human 4D MDS coordinates (32 unique schematic faces, Nosofsky 1991)",
    )
    plot_dimension_scatter_faces(
        human_mds,
        rotated,
        faces,
        face_numbers,
        output_folder / f"{model_slug}_vs_human_mds_scatter_faces",
        xlabel="Human MDS coordinate",
        ylabel=ylabel,
        suptitle=f"{model_name} (physical-rotated) vs. human 4D MDS coordinates (32 unique schematic faces, Nosofsky 1991)",
    )
    plot_dimension_scatter(
        phys_scaled,
        rotated,
        face_numbers,
        output_folder / f"{model_slug}_vs_physical_scatter",
        xlabel="Scaled physical coordinate (-EH, +ES, +NL, -MH)",
        ylabel=ylabel,
        suptitle=f"{model_name} (physical-rotated) 4D MDS vs. physical dimensions (32 unique schematic faces, Nosofsky 1991)",
    )
    plot_dimension_scatter_faces(
        phys_scaled,
        rotated,
        faces,
        face_numbers,
        output_folder / f"{model_slug}_vs_physical_scatter_faces",
        xlabel="Scaled physical coordinate (-EH, +ES, +NL, -MH)",
        ylabel=ylabel,
        suptitle=f"{model_name} (physical-rotated) 4D MDS vs. physical dimensions (32 unique schematic faces, Nosofsky 1991)",
    )
    plot_mds_space(
        rotated,
        faces,
        face_numbers,
        f"{model_name} 4D MDS solution rotated onto physical dimensions (stress = {stress:.3f})",
        output_folder / f"{model_slug}_mds_space",
    )
    plot_mds_space(
        human_mds,
        faces,
        face_numbers,
        "Human 4D MDS solution (Nosofsky 1991, Table A1)",
        output_folder / "human_mds_space",
    )

    print(f"\n4D Nonmetric MDS Stress ({model_name}): {stress:.4f}")
    print(f"\n{model_name} (Physical-Rotated) MDS vs. Physical Dimensions (-EH, +ES, +NL, -MH) Correlation Matrix:")
    print(vs_phys_oriented.round(3).to_string())
    print(f"\n{model_name} (Physical-Rotated) MDS vs. Human MDS Correlation Matrix:")
    print(vs_human.round(3).to_string())
    print("\nCorresponding Dimension Correlations (Diagonal):")
    for d, label in enumerate(DIM_LABELS):
        print(f"  {label}: vs. Physical r = {vs_phys_oriented.iloc[d, d]:.3f} | vs. Human MDS r = {vs_human.iloc[d, d]:.3f}")
    print(f"\nSaved MDS coordinates and plots to {output_folder}")


if __name__ == "__main__":
    main()

