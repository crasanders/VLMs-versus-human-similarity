"""Visualize a subsampled 8D human MDS solution against rock thumbnails, ratings, and published MDS.

Loads a saved 360x8 rotated MDS solution from `output/human_mds_subsampling/<mode>/mds_solutions/`
(picking one replicate uniformly at random if `--rep` is omitted) and saves 8 figure pairs (.png + .pdf):
  1. mds_space_2x2: 2x2 overview of Dim 1x2, 3x4, 5x6, 7x8 with 360 rock thumbnails
  2. mds_dim1x2, mds_dim3x4, mds_dim5x6, mds_dim7x8: individual 2D plane plots
  3. dims_vs_human_ratings_scatter: rotated Dim 1-6 vs. human perceptual dimension ratings
  4. dims_vs_human_ratings_heatmap: 8x6 correlation matrix between MDS dims and human ratings
  5. vs_published_human_mds: 2x4 Procrustes-aligned Dim 1-8 comparison vs. published human 8D MDS

Usage:
    python plot_subsampled_mds.py [--mode nonmetric|metric] [--pct 100] [--rep N]
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image
from scipy.linalg import orthogonal_procrustes
from scipy.spatial.distance import pdist

from make_jev_plots import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, style_axes

DEFAULT_INPUT_DIR = Path("output") / "human_mds_subsampling"
ROCKS_TABLE_PATH = Path("data") / "360_rocks_table.csv"
PUBLISHED_HUMAN_MDS_PATH = Path("rotated_mds_rocks360.txt")
THUMBNAIL_DIR = Path("data") / "Rocks Library" / "360 Rocks"

N_ROCKS = 360
N_DIMENSIONS = 8
THUMB_MAX_SIDE = 72

TARGET_DIM_COLUMNS = [
    "Lightness",
    "Grain Size",
    "Roughness",
    "Shininess",
    "Organization",
    "Munsell Chroma",
]

ROTATED_DIM_LABELS = [
    "Dim 1 (Lightness)",
    "Dim 2 (Grain Size)",
    "Dim 3 (Roughness)",
    "Dim 4 (Shininess)",
    "Dim 5 (Organization)",
    "Dim 6 (Munsell Chroma)",
    "Dim 7 (Unaligned)",
    "Dim 8 (Unaligned)",
]

HUMAN_MDS_LABELS = [
    "Dim 1 (Lightness)",
    "Dim 2 (Grain Size)",
    "Dim 3 (Roughness)",
    "Dim 4 (Shininess)",
    "Dim 5 (Organization)",
    "Dim 6 (Munsell Chroma)",
    "Dim 7 (Red/Green Hue)",
    "Dim 8 (Shape)",
]

DIVERGING = LinearSegmentedColormap.from_list("diverging", ["#a83232", "#f0efec", "#184f95"])
REP_FILE_PATTERN = re.compile(r"^mds_pct(\d{3})_rep(\d{2})\.csv$")


def save_fig(fig: plt.Figure, path_stem: Path) -> None:
    """Save a figure as both PNG (200 dpi) and PDF, then close it."""
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=200, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def select_replicate(mds_dir: Path, pct: int, rep: int | None) -> tuple[int, Path]:
    """Select an explicit replicate or pick one uniformly at random from available files."""
    candidates: dict[int, Path] = {}
    for path in sorted(mds_dir.glob(f"mds_pct{pct:03d}_rep*.csv")):
        match = REP_FILE_PATTERN.match(path.name)
        if match and int(match.group(1)) == pct:
            candidates[int(match.group(2))] = path
    if not candidates:
        raise FileNotFoundError(f"No MDS solutions found for pct={pct}% in {mds_dir}")

    if rep is not None:
        if rep not in candidates:
            raise FileNotFoundError(f"Replicate rep={rep} not found in {mds_dir} (available: {sorted(candidates)})")
        return rep, candidates[rep]

    rng = np.random.default_rng()
    chosen_rep = int(rng.choice(sorted(candidates)))
    return chosen_rep, candidates[chosen_rep]


def load_thumbnails(thumb_dir: Path, rock_ids: list[str], max_side: int = THUMB_MAX_SIDE) -> list[np.ndarray]:
    """Load and downsample transparent rock PNGs aligned to `rock_ids`."""
    png_map = {path.stem: path for path in thumb_dir.rglob("*.png")}
    missing = [rock for rock in rock_ids if rock not in png_map]
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} rock PNGs in {thumb_dir}; first missing: {missing[:5]}")

    thumbs: list[np.ndarray] = []
    for rock in rock_ids:
        with Image.open(png_map[rock]) as im:
            rgba = im.convert("RGBA")
            w, h = rgba.size
            scale = max_side / max(w, h)
            new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
            resized = rgba.resize(new_size, Image.Resampling.LANCZOS)
            thumbs.append(np.asarray(resized))
    return thumbs


def load_human_ratings(table_path: Path, rock_ids: list[str]) -> pd.DataFrame:
    """Load the 6 human perceptual rating columns aligned to `rock_ids`."""
    table = pd.read_csv(table_path, index_col=0)
    if len(table) != N_ROCKS:
        raise ValueError(f"Expected {N_ROCKS} rows in {table_path}, found {len(table)}")
    table["Rock"] = [f"{row.Category[0]}_{row.Subtype}_{int(row.Token):02d}" for row in table.itertuples()]
    ratings = table.set_index("Rock").loc[rock_ids, TARGET_DIM_COLUMNS].copy()
    if ratings.isna().any().any():
        raise ValueError(f"Found NaNs in {table_path} target ratings")
    return ratings


def add_rock_thumbnails(
    ax: plt.Axes,
    x: np.ndarray,
    y: np.ndarray,
    thumbs: list[np.ndarray],
    zoom: float,
    margin_frac: float = 0.08,
) -> None:
    """Plot downsampled rock PNGs at (x, y) coordinates with padded limits."""
    for thumb, x0, y0 in zip(thumbs, x, y):
        ab = AnnotationBbox(OffsetImage(thumb, zoom=zoom), (x0, y0), frameon=False, pad=0.0, zorder=3)
        ax.add_artist(ab)
    x_span = max(float(np.ptp(x)), 1e-6)
    y_span = max(float(np.ptp(y)), 1e-6)
    pad_x = margin_frac * x_span
    pad_y = margin_frac * y_span
    ax.set_xlim(float(np.min(x)) - pad_x, float(np.max(x)) + pad_x)
    ax.set_ylim(float(np.min(y)) - pad_y, float(np.max(y)) + pad_y)


def align_onto_published_human(coords: np.ndarray, published_human: np.ndarray) -> np.ndarray:
    """Orthogonally Procrustes-rotate `coords` onto `published_human` and match its norm/center."""
    hum_mean = published_human.mean(axis=0)
    hum_centered = published_human - hum_mean
    hum_norm = float(np.linalg.norm(hum_centered))

    centered = coords - coords.mean(axis=0)
    sol_norm = float(np.linalg.norm(centered))

    rotation, _ = orthogonal_procrustes(hum_centered / hum_norm, centered / sol_norm)
    aligned = (centered / sol_norm) @ rotation.T * hum_norm + hum_mean

    if not np.allclose(pdist(coords) / sol_norm, pdist(aligned) / hum_norm, rtol=1e-6, atol=1e-8):
        raise AssertionError("Procrustes alignment distorted relative pairwise distances")
    return aligned


def plot_mds_2x2(
    coords: np.ndarray,
    thumbs: list[np.ndarray],
    title: str,
    out_stem: Path,
) -> None:
    """Plot 2x2 overview of Dim 1x2, 3x4, 5x6, 7x8 with 360 rock thumbnails."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    for ax, d in zip(axes.flat, range(0, N_DIMENSIONS, 2)):
        style_axes(ax)
        add_rock_thumbnails(ax, coords[:, d], coords[:, d + 1], thumbs, zoom=0.32)
        ax.set_xlabel(ROTATED_DIM_LABELS[d], fontsize=11)
        ax.set_ylabel(ROTATED_DIM_LABELS[d + 1], fontsize=11)
        ax.set_title(f"{ROTATED_DIM_LABELS[d]} vs. {ROTATED_DIM_LABELS[d + 1]}", loc="left", fontsize=11.5)

    fig.suptitle(title, x=0.02, ha="left", fontsize=14, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save_fig(fig, out_stem)


def plot_mds_plane(
    coords: np.ndarray,
    thumbs: list[np.ndarray],
    d1: int,
    d2: int,
    title: str,
    out_stem: Path,
) -> None:
    """Plot a single 2D MDS plane with rock thumbnails."""
    fig, ax = plt.subplots(figsize=(11, 11))
    style_axes(ax)
    add_rock_thumbnails(ax, coords[:, d1], coords[:, d2], thumbs, zoom=0.42)
    ax.set_xlabel(ROTATED_DIM_LABELS[d1], fontsize=12)
    ax.set_ylabel(ROTATED_DIM_LABELS[d2], fontsize=12)
    ax.set_title(title, loc="left", fontsize=12.5, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, out_stem)


def plot_dims_vs_ratings_scatter(
    coords: np.ndarray,
    ratings: pd.DataFrame,
    thumbs: list[np.ndarray],
    title: str,
    out_stem: Path,
) -> None:
    """Plot rotated MDS Dim 1-6 against the 6 human perceptual dimension ratings."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
    badge = dict(boxstyle="round,pad=0.28", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8, alpha=0.92)

    for d, (ax, dim_name) in enumerate(zip(axes.flat, TARGET_DIM_COLUMNS)):
        style_axes(ax)
        x = coords[:, d]
        y = ratings[dim_name].to_numpy(dtype=float)
        r = float(np.corrcoef(x, y)[0, 1])

        add_rock_thumbnails(ax, x, y, thumbs, zoom=0.25)
        slope, intercept = np.polyfit(x, y, 1)
        x_line = np.linspace(float(x.min()), float(x.max()), 100)
        ax.plot(x_line, slope * x_line + intercept, color=TEXT_SECONDARY, linestyle="--", linewidth=1.3, zorder=2)

        ax.text(
            0.04,
            0.94,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10.5,
            fontweight="bold",
            color=TEXT_PRIMARY,
            bbox=badge,
            zorder=5,
        )
        ax.set_xlabel(ROTATED_DIM_LABELS[d], fontsize=10.5)
        ax.set_ylabel(f"Human {dim_name} Rating", fontsize=10.5)
        ax.set_title(f"{ROTATED_DIM_LABELS[d]}", loc="left", fontsize=11)

    fig.suptitle(title, x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save_fig(fig, out_stem)


def plot_dims_vs_ratings_heatmap(
    coords: np.ndarray,
    ratings: pd.DataFrame,
    title: str,
    out_stem: Path,
) -> pd.DataFrame:
    """Plot 8x6 correlation matrix between rotated MDS dimensions and human ratings."""
    corr_matrix = np.array(
        [
            [float(np.corrcoef(coords[:, d], ratings[col].to_numpy(dtype=float))[0, 1]) for col in TARGET_DIM_COLUMNS]
            for d in range(N_DIMENSIONS)
        ]
    )
    corr_df = pd.DataFrame(corr_matrix, index=ROTATED_DIM_LABELS, columns=TARGET_DIM_COLUMNS)

    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    im = ax.imshow(corr_matrix, cmap=DIVERGING, vmin=-1.0, vmax=1.0, aspect="auto")
    for (i, j), val in np.ndenumerate(corr_matrix):
        ax.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            fontsize=9.5,
            fontweight="bold" if i == j else "normal",
            color=SURFACE if abs(val) >= 0.6 else TEXT_PRIMARY,
        )
    ax.set_xticks(range(len(TARGET_DIM_COLUMNS)), TARGET_DIM_COLUMNS, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(range(N_DIMENSIONS), ROTATED_DIM_LABELS, fontsize=10)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Pearson r", color=TEXT_SECONDARY)
    ax.set_title(title, loc="left", fontsize=11.5, fontweight="bold", pad=10)
    fig.tight_layout()
    save_fig(fig, out_stem)
    return corr_df


def plot_vs_published_human_mds(
    aligned_coords: np.ndarray,
    published_human: np.ndarray,
    thumbs: list[np.ndarray],
    title: str,
    out_stem: Path,
) -> list[float]:
    """Plot Procrustes-aligned Dim 1-8 against published human 8D MDS (Fig. 5 style)."""
    fig, axes = plt.subplots(2, 4, figsize=(16, 8.2))
    badge = dict(boxstyle="round,pad=0.26", facecolor=SURFACE, edgecolor=GRID, linewidth=0.8, alpha=0.92)
    dim_rs: list[float] = []

    for d, ax in enumerate(axes.flat):
        style_axes(ax)
        x = aligned_coords[:, d]
        y = published_human[:, d]
        r = float(np.corrcoef(x, y)[0, 1])
        dim_rs.append(r)

        add_rock_thumbnails(ax, x, y, thumbs, zoom=0.22)
        lo = min(ax.get_xlim()[0], ax.get_ylim()[0])
        hi = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color=TEXT_SECONDARY, linestyle="--", linewidth=1.2, zorder=2)
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")

        ax.text(
            0.05,
            0.94,
            f"r = {r:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
            color=TEXT_PRIMARY,
            bbox=badge,
            zorder=5,
        )
        ax.set_title(HUMAN_MDS_LABELS[d], loc="left", fontsize=10.5)
        ax.set_xlabel("Subsampled MDS (Procrustes-aligned)", fontsize=9)
        if d % 4 == 0:
            ax.set_ylabel("Published Human 8D MDS", fontsize=9.5)

    fig.suptitle(title, x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, out_stem)
    return dim_rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["metric", "nonmetric"], default="nonmetric", help="MDS mode (default: nonmetric)")
    parser.add_argument("--pct", type=int, default=100, help="Sampling percentage (default: 100)")
    parser.add_argument("--rep", type=int, default=None, help="Replicate index (default: pick randomly)")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help=f"Subsampling root (default: {DEFAULT_INPUT_DIR})")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output folder for plots")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mode_dir = args.input_dir / args.mode
    mds_dir = mode_dir / "mds_solutions"
    rep, mds_path = select_replicate(mds_dir, args.pct, args.rep)
    output_dir = args.output_dir or (args.input_dir / "plots" / f"{args.mode}_pct{args.pct:03d}_rep{rep:02d}")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Selected MDS solution: mode={args.mode}, pct={args.pct}%, rep={rep:02d} ({mds_path})")

    mds_df = pd.read_csv(mds_path, index_col="Rock")
    rock_ids = list(mds_df.index)
    coords = mds_df[[f"Dim {d + 1}" for d in range(N_DIMENSIONS)]].to_numpy(dtype=float)

    summary_df = pd.read_csv(mode_dir / "summary.csv")
    row_match = summary_df[(summary_df["pct"] == args.pct) & (summary_df["rep"] == rep)]
    if len(row_match) != 1:
        raise ValueError(f"Expected 1 summary row for pct={args.pct}, rep={rep}; found {len(row_match)}")
    meta = row_match.iloc[0]

    ratings = load_human_ratings(ROCKS_TABLE_PATH, rock_ids)
    published_human = np.loadtxt(PUBLISHED_HUMAN_MDS_PATH)
    if published_human.shape != (N_ROCKS, N_DIMENSIONS):
        raise ValueError(f"Unexpected shape {published_human.shape} in {PUBLISHED_HUMAN_MDS_PATH}")

    thumbs = load_thumbnails(THUMBNAIL_DIR, rock_ids)

    mode_title = "Nonmetric" if args.mode == "nonmetric" else "Metric"
    subtitle = (
        f"{mode_title} 8D MDS ({args.pct}% human trials, rep {rep:02d}) — "
        f"Stress-1 = {meta['stress1_objective']:.4f}, "
        f"mean r (Dims 1–6) = {meta['mean_r_dims1_6']:.3f}"
    )

    plot_mds_2x2(coords, thumbs, subtitle, output_dir / "mds_space_2x2")
    for d in range(0, N_DIMENSIONS, 2):
        plane_title = (
            f"{mode_title} 8D MDS ({args.pct}%, rep {rep:02d}): "
            f"{ROTATED_DIM_LABELS[d]} vs. {ROTATED_DIM_LABELS[d + 1]} (Stress-1 = {meta['stress1_objective']:.4f})"
        )
        plot_mds_plane(coords, thumbs, d, d + 1, plane_title, output_dir / f"mds_dim{d + 1}x{d + 2}")

    plot_dims_vs_ratings_scatter(
        coords,
        ratings,
        thumbs,
        f"{mode_title} 8D MDS ({args.pct}%, rep {rep:02d}) rotated dims vs. human ratings (mean r = {meta['mean_r_dims1_6']:.3f})",
        output_dir / "dims_vs_human_ratings_scatter",
    )
    corr_df = plot_dims_vs_ratings_heatmap(
        coords,
        ratings,
        f"{mode_title} 8D MDS ({args.pct}%, rep {rep:02d}) vs. human dimension ratings",
        output_dir / "dims_vs_human_ratings_heatmap",
    )

    aligned_coords = align_onto_published_human(coords, published_human)
    dim_rs = plot_vs_published_human_mds(
        aligned_coords,
        published_human,
        thumbs,
        f"{mode_title} 8D MDS ({args.pct}%, rep {rep:02d}) Procrustes-aligned onto published human 8D MDS "
        f"(pairwise dist r = {meta['dist_r_vs_published_human']:.3f})",
        output_dir / "vs_published_human_mds",
    )

    print("Diagonal correlations vs. human dimension ratings:")
    for d, col in enumerate(TARGET_DIM_COLUMNS):
        print(f"  {ROTATED_DIM_LABELS[d]}: r = {corr_df.loc[ROTATED_DIM_LABELS[d], col]:.3f}")
    print("Aligned correlations vs. published human 8D MDS:")
    for d, r_val in enumerate(dim_rs):
        print(f"  {HUMAN_MDS_LABELS[d]}: r = {r_val:.3f}")
    print(f"Saved 8 plot pairs (.png and .pdf) to {output_dir}")


if __name__ == "__main__":
    main()
