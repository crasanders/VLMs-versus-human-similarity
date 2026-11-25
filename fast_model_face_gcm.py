"""Fit the Generalized Context Model (GCM; Nosofsky, 1991) to human face
classification data using the 4D MDS coordinates derived from Jev similarity
ratings (as well as the human 4D MDS coordinates from Table A1 for comparison).

Equations from Nosofsky (1991, Pages 5-9 & Appendix B, Page 23):
  - Eq. 1 (Weighted Euclidean distance):
      d_ij = c * [ sum_{m=1..4} w_m * |x_im - x_jm|^2 ]^(1/2),
      0 <= w_m <= 1, sum_{m=1..4} w_m = 1
  - Eq. 2 (Exponential decay similarity):
      s_ij = exp(-d_ij)
  - Eqs. 3-5 & Appendix B (Noisy exemplar activation & deterministic decision rule):
      Category 1 exemplars: Faces 1-5 (|C_1| = 5)
      Category 2 exemplars: Faces 6-10 (|C_2| = 5)
      E_j ~ N(s_ij, sigma^2) => E_{1,i} - E_{2,i} ~ N(sum_{j in C_1} s_ij - sum_{j in C_2} s_ij, 10 * sigma^2)
      P(C_1 | i) = Phi( (sum_{j in C_1} s_ij - sum_{j in C_2} s_ij - b) / (sqrt(10) * sigma) )
  - Footnote 3 (Maximum Likelihood objective):
      ln L = sum_{i=1..34} [ ln(N_i!) - ln(f_i1C!) - ln(f_i2C!) + f_i1C * ln(p_i1C) + f_i2C * ln(1 - p_i1C) ]
      where N_i = 160 for Exp 1A All Subjects (80 subjects x 2 transfer blocks)
      and N_i = 80 for Exp 1A Learners Only (40 subjects x 2 transfer blocks).
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
import pandas as pd
from PIL import Image
from scipy.optimize import differential_evolution, minimize
from scipy.special import gammaln, ndtr

from fast_model_face_mds import place_non_overlapping_labels

TABLE_A1_FILE = Path("data/nosofsky1991_table_a1.csv")
TABLE_1_FILE = Path("data/nosofsky1991_table_1.csv")
JEV_MDS_FILE = Path("output/face_mds/jev_mds.csv")
JEV_DIRECT_CLASS_FILE = Path("output/jev/face_classifications.csv")
FACE_IMAGE_DIR = Path("data/34 Faces")
OUTPUT_DIR = Path("output/face_mds")

CAT1_EXEMPLARS = [1, 2, 3, 4, 5]   # Category 1 training exemplars
CAT2_EXEMPLARS = [6, 7, 8, 9, 10]  # Category 2 training exemplars
DIM_NAMES = ["MDS 1 (EH)", "MDS 2 (ES)", "MDS 3 (NL)", "MDS 4 (MH)"]
DIM_SHORT = ["w1 (Eye Height)", "w2 (Eye Sep.)", "w3 (Nose Length)", "w4 (Mouth Height)"]

# Nosofsky (1991) Table 2 reported conjoint GCM parameters (for reference)
NOSOFSKY_TABLE2_PARAMS = {
    "All Subjects": {
        "sigma": 0.289,
        "c": 1.177,
        "w": np.array([0.19, 0.12, 0.25, 0.45]),
        "b": 0.181,
    },
    "Learners Only": {
        "sigma": 0.319,
        "c": 1.360,
        "w": np.array([0.29, 0.00, 0.64, 0.07]),
        "b": 0.162,
    },
}

# Plotting palette (matching fast_model_face_mds.py and fast_model_face_classification.py)
TEXT_PRIMARY = "#111827"
TEXT_SECONDARY = "#4B5563"
GRID = "#E5E7EB"
SURFACE = "#FFFFFF"
JEV_COLOR = "#2563EB"
HUMAN_COLOR = "#059669"
PAPER_COLOR = "#7C3AED"
CAT1_COLOR = "#2563EB"
CAT2_COLOR = "#D97706"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 9.5,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.8,
        "axes.facecolor": SURFACE,
        "figure.facecolor": SURFACE,
        "text.color": TEXT_PRIMARY,
        "axes.labelcolor": TEXT_SECONDARY,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def save_fig(fig: plt.Figure, path_stem: Path, dpi: int = 300) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    fig.savefig(path_stem.with_suffix(".pdf"), bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def load_34_face_coordinates() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Return DataFrame of all 34 faces, plus (34, 4) arrays for Jev MDS and Human MDS."""
    table_a1 = pd.read_csv(TABLE_A1_FILE)
    table_1 = pd.read_csv(TABLE_1_FILE)
    jev_mds_32 = pd.read_csv(JEV_MDS_FILE).set_index("Face")

    # Expand 32 unique faces to all 34 faces (Face 26 = Face 11, Face 30 = Face 18)
    jev_rows = []
    for face_id in table_a1["Face"].astype(int):
        lookup_id = 11 if face_id == 26 else (18 if face_id == 30 else face_id)
        row = jev_mds_32.loc[lookup_id]
        jev_rows.append(
            {
                "Face": face_id,
                "Jev_MDS_1": float(row["Jev_MDS_1"]),
                "Jev_MDS_2": float(row["Jev_MDS_2"]),
                "Jev_MDS_3": float(row["Jev_MDS_3"]),
                "Jev_MDS_4": float(row["Jev_MDS_4"]),
            }
        )
    jev_df = pd.DataFrame(jev_rows)
    df = table_a1.merge(jev_df, on="Face").merge(table_1, on="Face")

    jev_coords = df[["Jev_MDS_1", "Jev_MDS_2", "Jev_MDS_3", "Jev_MDS_4"]].to_numpy(dtype=float)
    human_coords = df[["MDS_1", "MDS_2", "MDS_3", "MDS_4"]].to_numpy(dtype=float)
    return df, jev_coords, human_coords


def gcm_predict_pc1(
    coords: np.ndarray,
    sigma: float,
    c: float,
    w: np.ndarray,
    b: float,
    cat1_idx: np.ndarray,
    cat2_idx: np.ndarray,
) -> np.ndarray:
    """Compute GCM predicted P(C1 | i) for all faces using Equations 1-5 and Appendix B."""
    w = np.asarray(w, dtype=float)
    w = np.clip(w, 0.0, None)
    w_sum = w.sum()
    if w_sum > 0:
        w = w / w_sum
    else:
        w = np.full(4, 0.25)

    # Pairwise squared differences: (N, N, 4)
    diff_sq = (coords[:, None, :] - coords[None, :, :]) ** 2
    # Weighted Euclidean distance (Eq. 1): d_ij = c * sqrt(sum_m w_m * |x_im - x_jm|^2)
    dist = c * np.sqrt(np.maximum(np.sum(diff_sq * w[None, None, :], axis=-1), 0.0))
    # Exponential decay similarity (Eq. 2): s_ij = exp(-d_ij)
    sim = np.exp(-dist)

    # Sum of similarities to Category 1 and Category 2 training exemplars
    sum_s1 = np.sum(sim[:, cat1_idx], axis=1)
    sum_s2 = np.sum(sim[:, cat2_idx], axis=1)

    # Appendix B: E_{1,i} - E_{2,i} ~ N(sum_s1 - sum_s2, (|C1| + |C2|) * sigma^2)
    n_exemplars = len(cat1_idx) + len(cat2_idx)  # 10
    denom = np.sqrt(n_exemplars) * max(sigma, 1e-8)
    z = (sum_s1 - sum_s2 - b) / denom
    return ndtr(z)


def gcm_neg_log_likelihood(
    pred_pc1: np.ndarray,
    obs_pc1: np.ndarray,
    n_trials: int,
) -> float:
    """Compute exact negative log-likelihood (-ln L) as defined in Nosofsky (1991) Footnote 3."""
    f1 = np.round(n_trials * obs_pc1)
    f2 = n_trials - f1
    eps = 1e-12
    p1 = np.clip(pred_pc1, eps, 1.0 - eps)
    log_comb = gammaln(n_trials + 1.0) - gammaln(f1 + 1.0) - gammaln(f2 + 1.0)
    ll = np.sum(log_comb + f1 * np.log(p1) + f2 * np.log(1.0 - p1))
    return float(-ll)


def compute_fit_metrics(
    pred_pc1: np.ndarray,
    obs_pc1: np.ndarray,
    n_trials: int,
) -> dict[str, float]:
    """Compute -ln L, SSE, %Var (1 - SSE/SST), Pearson r, and r^2."""
    neg_ll = gcm_neg_log_likelihood(pred_pc1, obs_pc1, n_trials)
    sse = float(np.sum((pred_pc1 - obs_pc1) ** 2))
    sst = float(np.sum((obs_pc1 - np.mean(obs_pc1)) ** 2))
    pct_var = 100.0 * (1.0 - sse / sst)
    r = float(np.corrcoef(pred_pc1, obs_pc1)[0, 1])
    return {
        "neg_ln_L": neg_ll,
        "SSE": sse,
        "pct_var": pct_var,
        "r": r,
        "r2_pct": 100.0 * (r ** 2),
    }


def fit_gcm(
    coords: np.ndarray,
    obs_pc1: np.ndarray,
    n_trials: int,
    cat1_idx: np.ndarray,
    cat2_idx: np.ndarray,
    seed: int = 42,
) -> dict:
    """Fit GCM parameters (sigma, c, w1, w2, w3, w4, b) by minimizing -ln L.

    Uses differential evolution over unconstrained/simplex-mapped parameters followed
    by multi-start SLSQP local polishing with explicit sum(w) = 1 constraint.
    """

    # For global search, parameterize as 7D vector: [sigma, c, u1, u2, u3, u4, b]
    # where w_m = u_m / sum(u). Fix u1..u4 in [0, 1] with sum > 0.
    def de_objective(theta: np.ndarray) -> float:
        sigma, c, u1, u2, u3, u4, b = theta
        u = np.array([u1, u2, u3, u4], dtype=float)
        u_sum = u.sum()
        if u_sum <= 1e-8:
            return 1e9
        w = u / u_sum
        pred = gcm_predict_pc1(coords, sigma, c, w, b, cat1_idx, cat2_idx)
        return gcm_neg_log_likelihood(pred, obs_pc1, n_trials)

    de_bounds = [
        (0.02, 5.0),   # sigma
        (0.05, 15.0),  # c
        (0.0, 1.0),    # u1
        (0.0, 1.0),    # u2
        (0.0, 1.0),    # u3
        (0.0, 1.0),    # u4
        (-2.0, 2.0),   # b
    ]

    de_res = differential_evolution(
        de_objective,
        bounds=de_bounds,
        seed=seed,
        maxiter=600,
        popsize=25,
        tol=1e-9,
        polish=False,
    )

    # Local constrained polishing via SLSQP with exact sum(w) = 1
    def slsqp_objective(theta: np.ndarray) -> float:
        sigma, c, w1, w2, w3, w4, b = theta
        w = np.array([w1, w2, w3, w4], dtype=float)
        pred = gcm_predict_pc1(coords, sigma, c, w, b, cat1_idx, cat2_idx)
        return gcm_neg_log_likelihood(pred, obs_pc1, n_trials)

    slsqp_bounds = [
        (0.005, 10.0),  # sigma
        (0.01, 25.0),   # c
        (0.0, 1.0),     # w1
        (0.0, 1.0),     # w2
        (0.0, 1.0),     # w3
        (0.0, 1.0),     # w4
        (-3.0, 3.0),    # b
    ]
    constraints = [{"type": "eq", "fun": lambda th: np.sum(th[2:6]) - 1.0}]

    # Normalize DE solution as first start
    u_de = np.clip(de_res.x[2:6], 0.0, None)
    w_de = u_de / u_de.sum() if u_de.sum() > 0 else np.full(4, 0.25)
    starts = [np.concatenate([[de_res.x[0], de_res.x[1]], w_de, [de_res.x[6]]])]

    # Additional starts (including Table 2 parameters and random perturbations)
    rng = np.random.default_rng(seed)
    starts.append(np.array([0.289, 1.177, 0.19, 0.12, 0.25, 0.45, 0.181]))
    starts.append(np.array([0.319, 1.360, 0.29, 0.00, 0.64, 0.07, 0.162]))
    starts.append(np.array([0.35, 3.5, 0.25, 0.25, 0.25, 0.25, 0.0]))
    for _ in range(15):
        rand_w = rng.dirichlet(np.ones(4))
        rand_sigma = rng.uniform(0.1, 1.0)
        rand_c = rng.uniform(0.5, 6.0)
        rand_b = rng.uniform(-0.4, 0.4)
        starts.append(np.concatenate([[rand_sigma, rand_c], rand_w, [rand_b]]))

    best_x = starts[0]
    best_fun = slsqp_objective(best_x)

    for x0 in starts:
        res = minimize(
            slsqp_objective,
            x0,
            method="SLSQP",
            bounds=slsqp_bounds,
            constraints=constraints,
            options={"ftol": 1e-12, "maxiter": 1000},
        )
        if res.fun < best_fun:
            best_fun = float(res.fun)
            best_x = res.x.copy()

    sigma, c = float(best_x[0]), float(best_x[1])
    w = np.clip(best_x[2:6], 0.0, None)
    w = w / w.sum()
    b = float(best_x[6])

    pred_pc1 = gcm_predict_pc1(coords, sigma, c, w, b, cat1_idx, cat2_idx)
    metrics = compute_fit_metrics(pred_pc1, obs_pc1, n_trials)

    return {
        "sigma": sigma,
        "c": c,
        "w": w,
        "b": b,
        "pred_pc1": pred_pc1,
        **metrics,
    }


def load_face_thumbnails(zoom: float = 0.095) -> dict[int, OffsetImage]:
    thumbs: dict[int, OffsetImage] = {}
    for face_num in range(1, 35):
        img_path = FACE_IMAGE_DIR / f"face_{face_num:02d}.png"
        img = Image.open(img_path).convert("RGBA")
        arr = np.array(img)
        rgb = arr[:, :, :3]
        is_bg = (rgb[:, :, 0] > 245) & (rgb[:, :, 1] > 245) & (rgb[:, :, 2] > 245)
        arr[is_bg, 3] = 0
        ys, xs = np.where(~is_bg)
        if len(xs) > 0 and len(ys) > 0:
            pad = 6
            y0, y1 = max(0, ys.min() - pad), min(arr.shape[0], ys.max() + pad + 1)
            x0, x1 = max(0, xs.min() - pad), min(arr.shape[1], xs.max() + pad + 1)
            arr = arr[y0:y1, x0:x1]
        thumbs[face_num] = OffsetImage(arr, zoom=zoom, resample=True)
    return thumbs


def plot_pred_vs_obs_scatter(
    df: pd.DataFrame,
    summary_df: pd.DataFrame,
    path: Path,
    use_thumbnails: bool = False,
) -> None:
    """2x2 scatterplot of GCM Predicted vs. Human Observed P(C1) as in Figure 5A of Nosofsky (1991)."""
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 10.2), dpi=300)
    thumbs = load_face_thumbnails(zoom=0.082) if use_thumbnails else {}

    panels = [
        ("Jev-MDS GCM", "All Subjects", "Jev_GCM_All_Pred_PC1", "Exp1A_All_Obs_PC1", axes[0, 0]),
        ("Jev-MDS GCM", "Learners Only", "Jev_GCM_Learners_Pred_PC1", "Exp1A_Learners_Obs_PC1", axes[0, 1]),
        ("Nosofsky (1991) Table 2 (Conjoint)", "All Subjects", "Nosofsky1991_Table2_All_Pred_PC1", "Exp1A_All_Obs_PC1", axes[1, 0]),
        ("Nosofsky (1991) Table 2 (Conjoint)", "Learners Only", "Nosofsky1991_Table2_Learners_Pred_PC1", "Exp1A_Learners_Obs_PC1", axes[1, 1]),
    ]

    lim = (-0.06, 1.06)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    faces = df["Face"].astype(int).tolist()

    for model_name, dataset_name, pred_col, obs_col, ax in panels:
        x = df[obs_col].to_numpy(dtype=float)  # Observed on x-axis or y-axis: match Fig 5A (Predicted on x, Observed on y)
        y = df[pred_col].to_numpy(dtype=float)

        row_meta = summary_df[
            (summary_df["Model"] == model_name) & (summary_df["Dataset"] == dataset_name)
        ].iloc[0]

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, linestyle="--", linewidth=0.5, color=GRID, alpha=0.7, zorder=0)
        ax.plot(lim, lim, linestyle="--", linewidth=1.0, color="#9CA3AF", zorder=1, label="y = x")

        for f_num, x0, y0 in zip(faces, x, y):
            is_cat1 = f_num in CAT1_EXEMPLARS
            is_cat2 = f_num in CAT2_EXEMPLARS
            color = CAT1_COLOR if is_cat1 else (CAT2_COLOR if is_cat2 else JEV_COLOR)
            z_base = 4 if (is_cat1 or is_cat2) else 3

            if use_thumbnails:
                img_copy = OffsetImage(thumbs[f_num].get_data(), zoom=thumbs[f_num].get_zoom(), resample=True)
                ab = AnnotationBbox(img_copy, (x0, y0), frameon=False, pad=0.0, zorder=z_base)
                ax.add_artist(ab)
                if is_cat1:
                    ax.scatter([x0], [y0], marker="o", s=560, facecolors="none", edgecolors=CAT1_COLOR, linewidths=1.35, zorder=z_base + 1)
                elif is_cat2:
                    ax.scatter([x0], [y0 - 0.008], marker="^", s=680, facecolors="none", edgecolors=CAT2_COLOR, linewidths=1.35, zorder=z_base + 1)
            else:
                marker = "o" if is_cat1 else ("^" if is_cat2 else "o")
                size = 55 if (is_cat1 or is_cat2) else 32
                ax.scatter([x0], [y0], marker=marker, s=size, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)

        labels = [str(f) for f in faces]
        colors = [
            CAT1_COLOR if f in CAT1_EXEMPLARS else (CAT2_COLOR if f in CAT2_EXEMPLARS else TEXT_PRIMARY)
            for f in faces
        ]
        weights = ["bold" if (f in CAT1_EXEMPLARS or f in CAT2_EXEMPLARS) else "normal" for f in faces]
        place_non_overlapping_labels(
            ax,
            x,
            y,
            labels=labels,
            colors=colors,
            weights=weights,
            lim=lim,
            fontsize=6.8,
            point_radius_norm=0.038 if use_thumbnails else 0.018,
            box_w_norm=0.045,
            box_h_norm=0.035,
            zorder=10,
        )

        if model_name == "Jev-MDS GCM" and dataset_name == "All Subjects" and not use_thumbnails:
            ax.scatter([], [], marker="o", s=45, color=CAT1_COLOR, label="Category 1 training (1–5)")
            ax.scatter([], [], marker="^", s=50, color=CAT2_COLOR, label="Category 2 training (6–10)")
            ax.scatter([], [], marker="o", s=32, color=JEV_COLOR, label="Transfer stimuli (11–34)")
            ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8)

        ax.set_title(
            f"{model_name} — {dataset_name}\n"
            f"-ln L = {row_meta['neg_ln_L']:.1f}, SSE = {row_meta['SSE']:.3f}, "
            f"%Var = {row_meta['pct_var']:.1f}% (r = {row_meta['r']:.3f})",
            loc="left",
            fontsize=9.8,
        )
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlabel(f"Human observed P(C₁) — {dataset_name}")
        ylabel_prefix = "Jev-MDS GCM" if model_name == "Jev-MDS GCM" else "Nosofsky (1991) Table 2"
        ax.set_ylabel(f"{ylabel_prefix} predicted P(C₁)")

    fig.suptitle(
        "GCM classification fits: Jev-MDS vs. Human-MDS predicted P(C₁) (Nosofsky 1991, Exp. 1A)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_fig(fig, path)


def plot_attention_weights(summary_df: pd.DataFrame, path: Path) -> None:
    """Grouped bar chart of GCM attention weights w1-w4 (as in Figure 6A of Nosofsky, 1991)."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), dpi=300, sharey=True)

    datasets = ["All Subjects", "Learners Only"]
    models = [
        ("Jev-MDS GCM", JEV_COLOR),
        ("Human-MDS GCM (Class. Fit)", HUMAN_COLOR),
        ("Nosofsky (1991) Table 2 (Conjoint)", PAPER_COLOR),
    ]

    x = np.arange(4)
    width = 0.25

    for ax_idx, (ax, ds) in enumerate(zip(axes, datasets)):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.5, color=GRID, alpha=0.7, zorder=0)
        ax.axhline(0.25, linestyle=":", linewidth=0.9, color="#9CA3AF", zorder=1, label="Uniform (0.25)" if ax_idx == 0 else None)

        for m_idx, (m_name, color) in enumerate(models):
            row = summary_df[(summary_df["Model"] == m_name) & (summary_df["Dataset"] == ds)].iloc[0]
            weights = [row["w1_EH"], row["w2_ES"], row["w3_NL"], row["w4_MH"]]
            offset = (m_idx - 1) * width
            bars = ax.bar(
                x + offset,
                weights,
                width=width,
                color=color,
                edgecolor=SURFACE,
                linewidth=0.6,
                label=m_name if ax_idx == 0 else None,
                zorder=3,
            )
            for bar, w_val in zip(bars, weights):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height() + 0.015,
                    f"{w_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color=TEXT_PRIMARY,
                )

        ax.set_title(f"Experiment 1A — {ds}", loc="left", fontsize=10.5, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(DIM_SHORT, fontsize=8.5)
        ax.set_ylim(0, 1.12)
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        if ax_idx == 0:
            ax.set_ylabel("Attention weight (wₘ)")
            ax.legend(loc="upper right", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8)

    fig.suptitle(
        "GCM fitted attention weights (w₁–w₄) across MDS representations (Nosofsky 1991, Fig. 6A)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save_fig(fig, path)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df, jev_coords, human_coords = load_34_face_coordinates()

    # Zero-based indices of Category 1 (Faces 1-5) and Category 2 (Faces 6-10)
    face_ids = df["Face"].astype(int).to_numpy()
    cat1_idx = np.where(np.isin(face_ids, CAT1_EXEMPLARS))[0]
    cat2_idx = np.where(np.isin(face_ids, CAT2_EXEMPLARS))[0]

    datasets = [
        ("All Subjects", "Exp1A_All_Obs_PC1", "Exp1A_All_Pred_PC1", 160),
        ("Learners Only", "Exp1A_Learners_Obs_PC1", "Exp1A_Learners_Pred_PC1", 80),
    ]

    summary_rows = []

    for ds_name, obs_col, paper_pred_col, n_trials in datasets:
        obs_pc1 = df[obs_col].to_numpy(dtype=float)
        prefix = "All" if ds_name == "All Subjects" else "Learners"

        # 1. Fit GCM on Jev-Derived 4D MDS
        jev_fit = fit_gcm(jev_coords, obs_pc1, n_trials, cat1_idx, cat2_idx, seed=42)
        df[f"Jev_GCM_{prefix}_Pred_PC1"] = np.round(jev_fit["pred_pc1"], 4)
        summary_rows.append(
            {
                "Model": "Jev-MDS GCM",
                "Dataset": ds_name,
                "N_trials_per_face": n_trials,
                "sigma": jev_fit["sigma"],
                "c": jev_fit["c"],
                "w1_EH": jev_fit["w"][0],
                "w2_ES": jev_fit["w"][1],
                "w3_NL": jev_fit["w"][2],
                "w4_MH": jev_fit["w"][3],
                "b": jev_fit["b"],
                "neg_ln_L": jev_fit["neg_ln_L"],
                "SSE": jev_fit["SSE"],
                "pct_var": jev_fit["pct_var"],
                "r": jev_fit["r"],
            }
        )

        # 2. Fit GCM on Human 4D MDS (classification-only ML fit)
        hum_fit = fit_gcm(human_coords, obs_pc1, n_trials, cat1_idx, cat2_idx, seed=42)
        df[f"Human_GCM_{prefix}_Pred_PC1"] = np.round(hum_fit["pred_pc1"], 4)
        summary_rows.append(
            {
                "Model": "Human-MDS GCM (Class. Fit)",
                "Dataset": ds_name,
                "N_trials_per_face": n_trials,
                "sigma": hum_fit["sigma"],
                "c": hum_fit["c"],
                "w1_EH": hum_fit["w"][0],
                "w2_ES": hum_fit["w"][1],
                "w3_NL": hum_fit["w"][2],
                "w4_MH": hum_fit["w"][3],
                "b": hum_fit["b"],
                "neg_ln_L": hum_fit["neg_ln_L"],
                "SSE": hum_fit["SSE"],
                "pct_var": hum_fit["pct_var"],
                "r": hum_fit["r"],
            }
        )

        # 3. Evaluate Nosofsky (1991) Table 2 Conjoint GCM parameters on Human 4D MDS
        t2 = NOSOFSKY_TABLE2_PARAMS[ds_name]
        t2_pred = gcm_predict_pc1(human_coords, t2["sigma"], t2["c"], t2["w"], t2["b"], cat1_idx, cat2_idx)
        t2_metrics = compute_fit_metrics(t2_pred, obs_pc1, n_trials)
        df[f"Nosofsky1991_Table2_{prefix}_Pred_PC1"] = np.round(t2_pred, 4)
        summary_rows.append(
            {
                "Model": "Nosofsky (1991) Table 2 (Conjoint)",
                "Dataset": ds_name,
                "N_trials_per_face": n_trials,
                "sigma": t2["sigma"],
                "c": t2["c"],
                "w1_EH": t2["w"][0],
                "w2_ES": t2["w"][1],
                "w3_NL": t2["w"][2],
                "w4_MH": t2["w"][3],
                "b": t2["b"],
                "neg_ln_L": t2_metrics["neg_ln_L"],
                "SSE": t2_metrics["SSE"],
                "pct_var": t2_metrics["pct_var"],
                "r": t2_metrics["r"],
            }
        )

    # Also include direct Jev classification predictions if available for reference
    if JEV_DIRECT_CLASS_FILE.exists():
        jev_direct = pd.read_csv(JEV_DIRECT_CLASS_FILE)[["Face", "Jev_PC1"]]
        df = df.merge(jev_direct, on="Face", how="left")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = OUTPUT_DIR / "gcm_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    pred_cols = [
        "Face",
        "Exp1A_All_Obs_PC1",
        "Jev_GCM_All_Pred_PC1",
        "Human_GCM_All_Pred_PC1",
        "Nosofsky1991_Table2_All_Pred_PC1",
        "Exp1A_All_Pred_PC1",
        "Exp1A_Learners_Obs_PC1",
        "Jev_GCM_Learners_Pred_PC1",
        "Human_GCM_Learners_Pred_PC1",
        "Nosofsky1991_Table2_Learners_Pred_PC1",
        "Exp1A_Learners_Pred_PC1",
    ]
    if "Jev_PC1" in df.columns:
        pred_cols.append("Jev_PC1")

    pred_df = df[pred_cols]
    pred_path = OUTPUT_DIR / "gcm_predictions.csv"
    pred_df.to_csv(pred_path, index=False)

    print("\n=== GCM Fit Summary (Nosofsky 1991 Equations 1-5 & Appendix B) ===")
    print(summary_df.to_string(index=False))
    print(f"\nSaved summary to {summary_path}")
    print(f"Saved per-face predictions to {pred_path}")

    plot_pred_vs_obs_scatter(df, summary_df, OUTPUT_DIR / "gcm_pred_vs_obs_scatter", use_thumbnails=False)
    plot_pred_vs_obs_scatter(df, summary_df, OUTPUT_DIR / "gcm_pred_vs_obs_scatter_faces", use_thumbnails=True)
    plot_attention_weights(summary_df, OUTPUT_DIR / "gcm_attention_weights")
    print(f"Saved plots to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
