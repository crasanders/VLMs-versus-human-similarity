"""Subsample human pairwise rock similarity ratings (10%-100%) and fit rotated 8D MDS spaces.

For each sampling percentage in 10%, 20%, ..., 100% and 10 replicates per percentage:
1. Sample individual pairwise trial ratings without replacement from the 253 valid
   (non-xlow) subjects in `data/Rocks-360 Similarity-Judgment Study/individual subject data`,
   enforcing at least 1 filled cell per row/column and a connected pair graph.
2. Fit an 8D weighted SMACOF MDS solution (metric and/or nonmetric with primary ties;
   by default nonmetric MDS is initialized from the converged metric MDS solution for
   each replicate, which yields lower Stress-1 and much higher replicate stability),
   assigning weight 0 to missing pairs.
3. Rotate the 8D solution via orthogonal Procrustes onto the 6 z-scored human
   dimension ratings from `data/360_rocks_table.csv` (padded with 2 zero columns),
   preserving all pairwise Euclidean distances.
4. Save rotated 360x8 coordinates and summary metrics for downstream GCM modeling.

Usage:
    python subsample_human_mds.py [--mds-mode metric|nonmetric|both] [--n-init 10] [--n-jobs -1]
"""

from __future__ import annotations

import os

# Pin BLAS/OpenMP threads before importing numpy/scipy/sklearn to avoid oversubscription
# when running joblib worker processes in parallel.
for _env_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_env_var, "1")

import argparse
import json
import re
import time
from pathlib import Path

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import isotonic_regression, minimize
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import pdist
from sklearn.metrics import euclidean_distances

SIMILARITY_DATA_DIR = Path("data") / "Rocks-360 Similarity-Judgment Study" / "individual subject data"
ROCKS_TABLE_PATH = Path("data") / "360_rocks_table.csv"
PUBLISHED_HUMAN_MDS_PATH = Path("rotated_mds_rocks360.txt")
DEFAULT_OUTPUT_DIR = Path("output") / "human_mds_subsampling"

N_ROCKS = 360
N_SUBTYPES = 30
N_TOKENS = 12
N_DIMENSIONS = 8
EXPECTED_SUBJECTS = 253
EXPECTED_TRIALS_PER_SUBJECT = 465

TARGET_DIM_COLUMNS = [
    "Lightness",
    "Grain Size",
    "Roughness",
    "Shininess",
    "Organization",
    "Munsell Chroma",
]

DIM_SUMMARY_COLS = [
    "r_lightness",
    "r_grain_size",
    "r_roughness",
    "r_shininess",
    "r_organization",
    "r_munsell_chroma",
]

SUMMARY_COLUMNS = [
    "mds_mode",
    "pct",
    "rep",
    "sample_seed",
    "n_ratings",
    "n_filled_pairs",
    "pct_filled_pairs",
    "min_row_degree",
    "is_connected",
    "resample_attempts",
    "n_init",
    "best_init",
    "n_iter",
    "stress1_objective",
    "stress1_metric",
    "stress1_nonmetric_primary",
    "dist_r_vs_published_human",
    *DIM_SUMMARY_COLS,
    "mean_r_dims1_6",
    "elapsed_sec",
    "mds_file",
]

VALID_SUBJECT_PATTERN = re.compile(r"^rocks_mds(\d+)\.txt$")
MODE_IDS = {"metric": 1, "nonmetric": 2, "ml": 3}


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    os.replace(tmp_path, path)


def atomic_write_dataframe(df: pd.DataFrame, path: Path, **to_csv_kwargs) -> None:
    """Write a DataFrame to CSV atomically via a same-directory temporary file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    df.to_csv(tmp_path, **to_csv_kwargs)
    os.replace(tmp_path, path)


def load_human_trials(sim_dir: Path) -> np.ndarray:
    """Load all valid (non-xlow) individual-subject pairwise ratings into (N, 3) array."""
    subject_files = sorted(
        (
            (int(match.group(1)), path)
            for path in sim_dir.iterdir()
            if path.is_file() and (match := VALID_SUBJECT_PATTERN.match(path.name))
        ),
        key=lambda item: item[0],
    )
    if len(subject_files) != EXPECTED_SUBJECTS:
        raise ValueError(
            f"Expected {EXPECTED_SUBJECTS} non-xlow subject files in {sim_dir}, found {len(subject_files)}"
        )

    chunks = []
    for _, path in subject_files:
        arr = np.loadtxt(path)
        if arr.shape != (EXPECTED_TRIALS_PER_SUBJECT, 5):
            raise ValueError(f"Unexpected shape {arr.shape} in {path}")
        sub1 = arr[:, 0].astype(int)
        sub2 = arr[:, 1].astype(int)
        tok1 = arr[:, 2].astype(int)
        tok2 = arr[:, 3].astype(int)
        ratings = arr[:, 4].astype(float)

        if np.any((sub1 < 1) | (sub1 > N_SUBTYPES) | (sub2 < 1) | (sub2 > N_SUBTYPES)):
            raise ValueError(f"Subtype index out of range [1, {N_SUBTYPES}] in {path}")
        if np.any((tok1 < 1) | (tok1 > N_TOKENS) | (tok2 < 1) | (tok2 > N_TOKENS)):
            raise ValueError(f"Token index out of range [1, {N_TOKENS}] in {path}")

        rock_i = N_TOKENS * (sub1 - 1) + (tok1 - 1)
        rock_j = N_TOKENS * (sub2 - 1) + (tok2 - 1)
        if np.any(rock_i == rock_j):
            raise ValueError(f"Found self-pair in {path}")

        rock_a = np.minimum(rock_i, rock_j)
        rock_b = np.maximum(rock_i, rock_j)
        chunks.append(np.column_stack([rock_a, rock_b, ratings]))

    trials = np.vstack(chunks)
    if np.any((trials[:, 0] < 0) | (trials[:, 1] >= N_ROCKS) | (trials[:, 0] >= trials[:, 1])):
        raise ValueError("Invalid canonical rock pair indices in loaded trials")
    return trials


def load_rock_table(table_path: Path) -> tuple[list[str], pd.DataFrame, np.ndarray]:
    """Load rock IDs, raw 6D human ratings, and z-scored 8D padded Procrustes target."""
    table = pd.read_csv(table_path, index_col=0)
    if len(table) != N_ROCKS:
        raise ValueError(f"Expected {N_ROCKS} rows in {table_path}, found {len(table)}")

    rock_ids = [f"{row.Category[0]}_{row.Subtype}_{int(row.Token):02d}" for row in table.itertuples()]
    ratings_df = table[TARGET_DIM_COLUMNS].copy()
    if ratings_df.isna().any().any():
        raise ValueError(f"Found NaNs in target rating columns of {table_path}")

    ratings_arr = ratings_df.to_numpy(dtype=float)
    target_z = (ratings_arr - ratings_arr.mean(axis=0)) / ratings_arr.std(axis=0, ddof=0)
    target_8d = np.pad(target_z, ((0, 0), (0, N_DIMENSIONS - target_z.shape[1])))
    return rock_ids, ratings_df, target_8d


def subsample_similarity_matrix(
    trials: np.ndarray,
    pct: int,
    sample_seed: int,
    n_rocks: int = N_ROCKS,
    max_attempts: int = 1000,
) -> dict:
    """Subsample individual trial ratings without replacement and average per pair."""
    total_trials = len(trials)
    n_ratings = int(round(total_trials * pct / 100.0))
    if not (1 <= n_ratings <= total_trials):
        raise ValueError(f"Invalid n_ratings={n_ratings} for pct={pct}")

    rng = np.random.default_rng(sample_seed)
    n_attempts = 1 if n_ratings == total_trials else max_attempts

    for attempt in range(1, n_attempts + 1):
        if n_ratings == total_trials:
            sub = trials
        else:
            idx = rng.choice(total_trials, size=n_ratings, replace=False)
            sub = trials[idx]

        sums = np.zeros((n_rocks, n_rocks), dtype=float)
        cnts = np.zeros((n_rocks, n_rocks), dtype=int)
        rock_a = sub[:, 0].astype(int)
        rock_b = sub[:, 1].astype(int)
        np.add.at(sums, (rock_a, rock_b), sub[:, 2])
        np.add.at(cnts, (rock_a, rock_b), 1)

        sums = sums + sums.T
        cnts = cnts + cnts.T
        mask = cnts > 0
        min_row_degree = int(mask.sum(axis=1).min())
        if min_row_degree < 1:
            continue

        n_comp, _ = connected_components(mask, directed=False)
        if n_comp != 1:
            continue

        mean_sim = np.zeros((n_rocks, n_rocks), dtype=float)
        mean_sim[mask] = sums[mask] / cnts[mask]
        n_filled_pairs = int(np.triu(mask, 1).sum())
        total_pairs = n_rocks * (n_rocks - 1) // 2
        return {
            "mean_sim": mean_sim,
            "cnts": cnts,
            "mask": mask,
            "n_ratings": n_ratings,
            "n_filled_pairs": n_filled_pairs,
            "pct_filled_pairs": 100.0 * n_filled_pairs / total_pairs,
            "min_row_degree": min_row_degree,
            "is_connected": True,
            "resample_attempts": attempt,
        }

    raise RuntimeError(
        f"Failed to sample a connected matrix with min_row_degree >= 1 at pct={pct}% "
        f"within {max_attempts} attempts"
    )


def compute_metric_stress1(d_obs: np.ndarray, dissim_obs: np.ndarray) -> float:
    """Compute scale-invariant Kruskal Stress-1 for metric MDS over observed pairs."""
    denom_dissim = float(np.dot(dissim_obs, dissim_obs))
    alpha = float(np.dot(d_obs, dissim_obs) / denom_dissim) if denom_dissim > 0 else 1.0
    denom_d = max(float(np.dot(d_obs, d_obs)), 1e-12)
    return float(np.sqrt(np.sum((d_obs - alpha * dissim_obs) ** 2) / denom_d))


def compute_nonmetric_primary_stress1(
    d_obs: np.ndarray, groups: list[np.ndarray]
) -> tuple[float, np.ndarray]:
    """Compute Kruskal Stress-1 and disparities under primary tie handling."""
    order = np.concatenate([g[np.argsort(d_obs[g])] if len(g) > 1 else g for g in groups])
    dhat_sorted = isotonic_regression(d_obs[order]).x
    dhat_obs = np.empty_like(d_obs)
    dhat_obs[order] = dhat_sorted
    denom_d = max(float(np.dot(d_obs, d_obs)), 1e-12)
    stress1 = float(np.sqrt(np.sum((d_obs - dhat_obs) ** 2) / denom_d))
    return stress1, dhat_obs


def _smacof_guttman_step(
    X: np.ndarray,
    V_pinv: np.ndarray,
    obs_i: np.ndarray,
    obs_j: np.ndarray,
    d_obs: np.ndarray,
    dhat_obs: np.ndarray,
) -> np.ndarray:
    """Perform one weighted SMACOF Guttman majorization update and center X."""
    n = X.shape[0]
    ratio = dhat_obs / np.maximum(d_obs, 1e-12)
    B = np.zeros((n, n), dtype=float)
    B[obs_i, obs_j] = -ratio
    B[obs_j, obs_i] = -ratio
    B[np.arange(n), np.arange(n)] = -B.sum(axis=1)
    X_new = V_pinv @ (B @ X)
    X_new -= X_new.mean(axis=0)
    return X_new


def fit_weighted_smacof(
    mean_sim: np.ndarray,
    mask: np.ndarray,
    mode: str,
    n_components: int = N_DIMENSIONS,
    n_init: int = 10,
    max_iter: int = 1000,
    eps: float = 1e-6,
    metric_warmup: int = 50,
    mds_seed: int = 0,
    init_X: np.ndarray | None = None,
) -> dict:
    """Fit 8D weighted metric or nonmetric SMACOF with missing-pair weights set to 0."""
    if mode not in MODE_IDS:
        raise ValueError(f"Unsupported MDS mode: {mode}")

    n = mean_sim.shape[0]
    W = mask.astype(float)
    np.fill_diagonal(W, 0.0)
    V = -W
    V[np.arange(n), np.arange(n)] = W.sum(axis=1)
    V_pinv = np.linalg.pinv(V)

    obs_i, obs_j = np.where(np.triu(mask, 1))
    obs_sim = mean_sim[obs_i, obs_j]
    n_obs = len(obs_i)

    # Metric target uses 9 - sim (mean_sim == 9.0 gives 0 dissimilarity with weight 1.0).
    dissim_metric = 9.0 - obs_sim
    metric_norm = np.sqrt(np.sum(dissim_metric ** 2))
    dhat_metric = dissim_metric * (np.sqrt(n_obs) / metric_norm) if metric_norm > 0 else dissim_metric.copy()

    # Nonmetric target uses 10 - sim (strictly positive [1, 9]) and primary tie groups.
    dissim_nonmetric = 10.0 - obs_sim
    _, counts_u = np.unique(dissim_nonmetric, return_counts=True)
    base_order = np.argsort(dissim_nonmetric, kind="mergesort")
    groups = np.split(base_order, np.cumsum(counts_u)[:-1])

    if init_X is not None:
        init_configs: list[tuple[int, np.ndarray]] = [(0, init_X.astype(float, copy=True))]
    else:
        init_seeds = np.random.SeedSequence(mds_seed).spawn(n_init)
        init_configs = [
            (init_idx, np.random.default_rng(child_seed).normal(size=(n, n_components)))
            for init_idx, child_seed in enumerate(init_seeds)
        ]

    best_overall_X: np.ndarray | None = None
    best_overall_stress = np.inf
    best_overall_init = -1
    best_overall_iter = 0

    for init_idx, X in init_configs:
        X = X - X.mean(axis=0)

        if init_X is None and mode == "nonmetric" and metric_warmup > 0:
            for _ in range(metric_warmup):
                d_obs = euclidean_distances(X)[obs_i, obs_j]
                X = _smacof_guttman_step(X, V_pinv, obs_i, obs_j, d_obs, dhat_metric)

        best_run_X = X.copy()
        best_run_stress = np.inf
        best_run_iter = 0
        old_stress: float | None = None

        for it in range(max_iter):
            d_obs = euclidean_distances(X)[obs_i, obs_j]
            if mode == "metric":
                stress1 = compute_metric_stress1(d_obs, dissim_metric)
                dhat_step = dhat_metric
            else:
                stress1, dhat_obs = compute_nonmetric_primary_stress1(d_obs, groups)
                norm_dhat = float(np.sqrt(np.sum(dhat_obs ** 2)))
                dhat_step = dhat_obs * (np.sqrt(n_obs) / norm_dhat) if norm_dhat > 0 else dhat_obs

            if stress1 < best_run_stress:
                best_run_stress = stress1
                best_run_X = X.copy()
                best_run_iter = it

            if old_stress is not None and abs(old_stress - stress1) < eps and it > 5:
                break
            old_stress = stress1

            X = _smacof_guttman_step(X, V_pinv, obs_i, obs_j, d_obs, dhat_step)
        else:
            d_obs = euclidean_distances(X)[obs_i, obs_j]
            if mode == "metric":
                stress1 = compute_metric_stress1(d_obs, dissim_metric)
            else:
                stress1, _ = compute_nonmetric_primary_stress1(d_obs, groups)
            if stress1 < best_run_stress:
                best_run_stress = stress1
                best_run_X = X.copy()
                best_run_iter = max_iter

        if best_run_stress < best_overall_stress:
            best_overall_stress = best_run_stress
            best_overall_X = best_run_X
            best_overall_init = init_idx
            best_overall_iter = best_run_iter

    assert best_overall_X is not None
    final_d_obs = euclidean_distances(best_overall_X)[obs_i, obs_j]
    stress1_metric = compute_metric_stress1(final_d_obs, dissim_metric)
    stress1_nonmetric_primary, _ = compute_nonmetric_primary_stress1(final_d_obs, groups)

    return {
        "X": best_overall_X,
        "stress1_objective": float(best_overall_stress),
        "stress1_metric": float(stress1_metric),
        "stress1_nonmetric_primary": float(stress1_nonmetric_primary),
        "best_init": int(best_overall_init),
        "n_iter": int(best_overall_iter),
    }


def fit_ml_mds(
    mean_sim: np.ndarray,
    cnts: np.ndarray,
    mask: np.ndarray,
    n_components: int = N_DIMENSIONS,
    n_init: int = 5,
    max_iter: int = 1000,
    mds_seed: int = 0,
) -> dict:
    """Fit 8D Maximum-Likelihood MDS (Nosofsky et al., 2018, Eqs. 3-4).

    Under i.i.d. Gaussian trial noise with linear similarity-distance mapping
    s_hat_ij = u - v * d_ij (absorbing v > 0 into the coordinate scale of X),
    minimizing individual-trial SSD is exact count-weighted least squares:
        min_{X, u}  0.5 * sum_{i < j} n_ij * ((u - d_ij) - s_bar_ij)^2
    solved with L-BFGS-B using the exact analytic gradient.
    """
    n = mean_sim.shape[0]
    obs_i, obs_j = np.where(np.triu(mask, 1))
    obs_sim = mean_sim[obs_i, obs_j]
    w_obs = cnts[obs_i, obs_j].astype(float)

    dissim_metric = 9.0 - obs_sim
    dissim_nonmetric = 10.0 - obs_sim
    _, counts_u = np.unique(dissim_nonmetric, return_counts=True)
    base_order = np.argsort(dissim_nonmetric, kind="mergesort")
    groups = np.split(base_order, np.cumsum(counts_u)[:-1])

    def obj_and_grad(params: np.ndarray) -> tuple[float, np.ndarray]:
        u = params[0]
        X = params[1:].reshape(n, n_components)
        D = euclidean_distances(X)
        d_obs = D[obs_i, obs_j]
        resid = (u - d_obs) - obs_sim
        loss = 0.5 * float(np.sum(w_obs * resid ** 2))
        grad_u = float(np.sum(w_obs * resid))

        m_off = (w_obs * resid) / np.maximum(d_obs, 1e-12)
        M = np.zeros((n, n), dtype=float)
        M[obs_i, obs_j] = m_off
        M[obs_j, obs_i] = m_off
        M[np.arange(n), np.arange(n)] = -M.sum(axis=1)
        grad_X = M @ X
        return loss, np.concatenate([[grad_u], grad_X.ravel()])

    init_seeds = np.random.SeedSequence(mds_seed).spawn(n_init)
    best_X: np.ndarray | None = None
    best_u = 9.0
    best_loss = np.inf
    best_init = -1
    best_iter = 0

    for init_idx, child_seed in enumerate(init_seeds):
        rng = np.random.default_rng(child_seed)
        X0 = rng.normal(scale=2.0, size=(n, n_components))
        X0 -= X0.mean(axis=0)
        p0 = np.concatenate([[9.0], X0.ravel()])
        res = minimize(
            obj_and_grad,
            p0,
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": max_iter, "ftol": 1e-12, "gtol": 1e-6},
        )
        if res.fun < best_loss:
            best_loss = float(res.fun)
            best_u = float(res.x[0])
            best_X = res.x[1:].reshape(n, n_components)
            best_X = best_X - best_X.mean(axis=0)
            best_init = init_idx
            best_iter = int(res.nit)

    assert best_X is not None
    final_d_obs = euclidean_distances(best_X)[obs_i, obs_j]
    resid_final = (best_u - final_d_obs) - obs_sim
    denom_d = max(float(np.sum(w_obs * final_d_obs ** 2)), 1e-12)
    stress1_ml = float(np.sqrt(np.sum(w_obs * resid_final ** 2) / denom_d))
    stress1_metric = compute_metric_stress1(final_d_obs, dissim_metric)
    stress1_nonmetric_primary, _ = compute_nonmetric_primary_stress1(final_d_obs, groups)

    return {
        "X": best_X,
        "stress1_objective": stress1_ml,
        "stress1_metric": float(stress1_metric),
        "stress1_nonmetric_primary": float(stress1_nonmetric_primary),
        "best_init": int(best_init),
        "n_iter": int(best_iter),
    }


def rotate_orthogonal_procrustes(solution: np.ndarray, target_8d: np.ndarray) -> np.ndarray:
    """Orthogonally rotate/reflect `solution` onto `target_8d`, preserving distances."""
    target = target_8d - target_8d.mean(axis=0)
    centered = solution - solution.mean(axis=0)
    rotation, _ = orthogonal_procrustes(
        target / np.linalg.norm(target),
        centered / np.linalg.norm(centered),
    )
    rotated = centered @ rotation.T
    if not np.allclose(pdist(solution), pdist(rotated), rtol=1e-6, atol=1e-8):
        raise AssertionError("Orthogonal Procrustes rotation altered pairwise distances")
    return rotated


def make_seed(base_seed: int, *parts: int) -> int:
    """Derive a reproducible 32-bit unsigned integer seed from hierarchical indices."""
    return int(np.random.SeedSequence([base_seed, *parts]).generate_state(1)[0])


def run_single_replicate(
    trials: np.ndarray,
    rock_ids: list[str],
    ratings_arr: np.ndarray,
    target_8d: np.ndarray,
    published_human_pdist: np.ndarray,
    output_dir: Path,
    mode: str,
    pct: int,
    rep: int,
    base_seed: int,
    n_init: int,
    max_iter: int,
    eps: float,
    metric_warmup: int,
    overwrite: bool,
    init_from_metric: bool = True,
) -> tuple[dict, bool]:
    """Run (or load cached) subsampling, weighted MDS, Procrustes rotation, and export."""
    mode_dir = output_dir / mode
    csv_path = mode_dir / "mds_solutions" / f"mds_pct{pct:03d}_rep{rep:02d}.csv"
    meta_path = mode_dir / "meta" / f"meta_pct{pct:03d}_rep{rep:02d}.json"

    if not overwrite and csv_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        return meta, True

    t0 = time.perf_counter()
    sample_seed = make_seed(base_seed, pct, rep)
    mds_seed = make_seed(base_seed, pct, rep, MODE_IDS[mode])

    sample = subsample_similarity_matrix(trials, pct=pct, sample_seed=sample_seed)
    init_X: np.ndarray | None = None
    effective_n_init = n_init
    if mode == "ml":
        mds_res = fit_ml_mds(
            mean_sim=sample["mean_sim"],
            cnts=sample["cnts"],
            mask=sample["mask"],
            n_components=N_DIMENSIONS,
            n_init=n_init,
            max_iter=max_iter,
            mds_seed=mds_seed,
        )
    else:
        if mode == "nonmetric" and init_from_metric:
            metric_csv = output_dir / "metric" / "mds_solutions" / f"mds_pct{pct:03d}_rep{rep:02d}.csv"
            if metric_csv.exists():
                metric_df = pd.read_csv(metric_csv, index_col=0)
                init_X = metric_df[[f"Dim {d + 1}" for d in range(N_DIMENSIONS)]].to_numpy(dtype=float)
            else:
                metric_seed = make_seed(base_seed, pct, rep, MODE_IDS["metric"])
                metric_res = fit_weighted_smacof(
                    mean_sim=sample["mean_sim"],
                    mask=sample["mask"],
                    mode="metric",
                    n_components=N_DIMENSIONS,
                    n_init=n_init,
                    max_iter=max_iter,
                    eps=eps,
                    metric_warmup=0,
                    mds_seed=metric_seed,
                )
                init_X = metric_res["X"]
            effective_n_init = 1

        mds_res = fit_weighted_smacof(
            mean_sim=sample["mean_sim"],
            mask=sample["mask"],
            mode=mode,
            n_components=N_DIMENSIONS,
            n_init=effective_n_init,
            max_iter=max_iter,
            eps=eps,
            metric_warmup=metric_warmup,
            mds_seed=mds_seed,
            init_X=init_X,
        )
    rotated = rotate_orthogonal_procrustes(mds_res["X"], target_8d)

    mds_df = pd.DataFrame(
        rotated,
        index=rock_ids,
        columns=[f"Dim {d + 1}" for d in range(N_DIMENSIONS)],
    )
    atomic_write_dataframe(mds_df, csv_path, index_label="Rock")

    dim_corrs = {
        col: float(np.corrcoef(rotated[:, d], ratings_arr[:, d])[0, 1])
        for d, col in enumerate(DIM_SUMMARY_COLS)
    }
    dist_r = float(np.corrcoef(pdist(rotated), published_human_pdist)[0, 1])
    elapsed = time.perf_counter() - t0

    meta = {
        "mds_mode": mode,
        "pct": int(pct),
        "rep": int(rep),
        "sample_seed": int(sample_seed),
        "n_ratings": int(sample["n_ratings"]),
        "n_filled_pairs": int(sample["n_filled_pairs"]),
        "pct_filled_pairs": float(sample["pct_filled_pairs"]),
        "min_row_degree": int(sample["min_row_degree"]),
        "is_connected": bool(sample["is_connected"]),
        "resample_attempts": int(sample["resample_attempts"]),
        "n_init": int(effective_n_init),
        "best_init": int(mds_res["best_init"]),
        "n_iter": int(mds_res["n_iter"]),
        "stress1_objective": float(mds_res["stress1_objective"]),
        "stress1_metric": float(mds_res["stress1_metric"]),
        "stress1_nonmetric_primary": float(mds_res["stress1_nonmetric_primary"]),
        "dist_r_vs_published_human": dist_r,
        **dim_corrs,
        "mean_r_dims1_6": float(np.mean(list(dim_corrs.values()))),
        "elapsed_sec": float(elapsed),
        "mds_file": csv_path.as_posix(),
    }
    atomic_write_text(meta_path, json.dumps(meta, indent=2))
    return meta, False


def format_grouped_summary(summary_df: pd.DataFrame) -> str:
    """Format mean +- SD summary table grouped by (mds_mode, pct)."""
    metrics = [
        ("n_filled_pairs", ".0f"),
        ("pct_filled_pairs", ".1f"),
        ("min_row_degree", ".1f"),
        ("stress1_objective", ".4f"),
        ("dist_r_vs_published_human", ".4f"),
        ("mean_r_dims1_6", ".4f"),
    ]
    rows = []
    for (mode, pct), grp in summary_df.groupby(["mds_mode", "pct"], sort=True):
        row = {"mds_mode": mode, "pct": pct, "n_reps": len(grp)}
        for col, fmt in metrics:
            mean_val = grp[col].mean()
            std_val = grp[col].std(ddof=1) if len(grp) > 1 else 0.0
            row[col] = f"{mean_val:{fmt}} +- {std_val:{fmt}}"
        rows.append(row)
    return pd.DataFrame(rows).to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mds-mode",
        choices=["metric", "nonmetric", "ml", "both", "all"],
        default="all",
        help="MDS formulation to run: metric, nonmetric, ml (Nosofsky et al. 2018 ML-MDS), both (metric+nonmetric), or all (default: all)",
    )
    parser.add_argument(
        "--percentages",
        type=int,
        nargs="+",
        default=list(range(10, 101, 10)),
        help="Sampling percentages in [1, 100] (default: 10 20 ... 100)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=10,
        help="Number of replicates per sampling percentage (default: 10)",
    )
    parser.add_argument(
        "--n-init",
        type=int,
        default=10,
        help="Random SMACOF restarts per replicate (default: 10)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=1000,
        help="Maximum SMACOF iterations per restart (default: 1000)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-6,
        help="Stress-1 convergence tolerance (default: 1e-6)",
    )
    parser.add_argument(
        "--metric-warmup",
        type=int,
        default=50,
        help="Weighted metric SMACOF warm-up iterations for nonmetric mode (default: 50)",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=-1,
        help="Number of joblib parallel workers (default: -1 for all cores)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base random seed for subsampling and MDS initializations (default: 42)",
    )
    parser.add_argument(
        "--sim-dir",
        type=Path,
        default=SIMILARITY_DATA_DIR,
        help=f"Directory with individual subject similarity files (default: {SIMILARITY_DATA_DIR})",
    )
    parser.add_argument(
        "--rocks-table",
        type=Path,
        default=ROCKS_TABLE_PATH,
        help=f"Path to 360_rocks_table.csv (default: {ROCKS_TABLE_PATH})",
    )
    parser.add_argument(
        "--published-human-mds",
        type=Path,
        default=PUBLISHED_HUMAN_MDS_PATH,
        help=f"Path to published human 8D MDS solution (default: {PUBLISHED_HUMAN_MDS_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output root directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--no-init-from-metric",
        action="store_true",
        help="Initialize nonmetric MDS from random starts instead of converged metric MDS solutions",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Recompute even if cached CSV and meta JSON already exist",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for pct in args.percentages:
        if not (1 <= pct <= 100):
            raise ValueError(f"Percentages must be in [1, 100], got {pct}")
    if args.reps < 1 or args.n_init < 1 or args.max_iter < 1:
        raise ValueError("reps, n_init, and max_iter must be >= 1")

    if args.mds_mode == "both":
        modes = ["metric", "nonmetric"]
    elif args.mds_mode == "all":
        modes = ["metric", "nonmetric", "ml"]
    else:
        modes = [args.mds_mode]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tasks = [(mode, pct, rep) for mode in modes for pct in args.percentages for rep in range(args.reps)]
    t_start = time.perf_counter()
    results = []
    pending_tasks = []

    if not args.overwrite:
        for mode, pct, rep in tasks:
            mode_dir = args.output_dir / mode
            csv_path = mode_dir / "mds_solutions" / f"mds_pct{pct:03d}_rep{rep:02d}.csv"
            meta_path = mode_dir / "meta" / f"meta_pct{pct:03d}_rep{rep:02d}.json"
            if csv_path.exists() and meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                results.append(meta)
                print(
                    f"[{len(results):3d}/{len(tasks):3d}] cached | {meta['mds_mode']:9s} pct={meta['pct']:3d}% "
                    f"rep={meta['rep']:02d} | filled={meta['pct_filled_pairs']:5.1f}% "
                    f"deg_min={meta['min_row_degree']:3d} | stress1={meta['stress1_objective']:.4f} "
                    f"dist_r={meta['dist_r_vs_published_human']:.4f} mean_r6={meta['mean_r_dims1_6']:.3f} "
                    f"({meta['elapsed_sec']:.1f}s)",
                    flush=True,
                )
            else:
                pending_tasks.append((mode, pct, rep))
    else:
        pending_tasks = list(tasks)

    if pending_tasks:
        print(f"Loading individual subject trials from {args.sim_dir}...")
        trials = load_human_trials(args.sim_dir)
        rock_ids, ratings_df, target_8d = load_rock_table(args.rocks_table)
        ratings_arr = ratings_df.to_numpy(dtype=float)

        published_human_mds = np.loadtxt(args.published_human_mds)
        if published_human_mds.shape != (N_ROCKS, N_DIMENSIONS):
            raise ValueError(
                f"Expected shape {(N_ROCKS, N_DIMENSIONS)} in {args.published_human_mds}, "
                f"got {published_human_mds.shape}"
            )
        published_human_pdist = pdist(published_human_mds)

        print(
            f"Loaded {len(trials):,} trials across {EXPECTED_SUBJECTS} non-xlow subjects. "
            f"Running {len(pending_tasks)} MDS tasks (modes={modes}, percentages={args.percentages}, "
            f"reps={args.reps}, n_init={args.n_init}, max_iter={args.max_iter}, n_jobs={args.n_jobs})..."
        )

        init_from_metric = not args.no_init_from_metric
        # Run metric before nonmetric when initializing nonmetric from converged metric solutions
        task_batches = (
            [
                [t for t in pending_tasks if t[0] != "nonmetric"],
                [t for t in pending_tasks if t[0] == "nonmetric"],
            ]
            if ("metric" in modes and "nonmetric" in modes and init_from_metric)
            else [pending_tasks]
        )

        for batch in task_batches:
            if not batch:
                continue
            gen = Parallel(n_jobs=args.n_jobs, return_as="generator_unordered")(
                delayed(run_single_replicate)(
                    trials=trials,
                    rock_ids=rock_ids,
                    ratings_arr=ratings_arr,
                    target_8d=target_8d,
                    published_human_pdist=published_human_pdist,
                    output_dir=args.output_dir,
                    mode=mode,
                    pct=pct,
                    rep=rep,
                    base_seed=args.seed,
                    n_init=args.n_init,
                    max_iter=args.max_iter,
                    eps=args.eps,
                    metric_warmup=args.metric_warmup,
                    overwrite=args.overwrite,
                    init_from_metric=init_from_metric,
                )
                for mode, pct, rep in batch
            )

            for meta, from_cache in gen:
                results.append(meta)
                tag = "cached" if from_cache else "done"
                print(
                    f"[{len(results):3d}/{len(tasks):3d}] {tag:6s} | {meta['mds_mode']:9s} pct={meta['pct']:3d}% "
                    f"rep={meta['rep']:02d} | filled={meta['pct_filled_pairs']:5.1f}% "
                    f"deg_min={meta['min_row_degree']:3d} | stress1={meta['stress1_objective']:.4f} "
                    f"dist_r={meta['dist_r_vs_published_human']:.4f} mean_r6={meta['mean_r_dims1_6']:.3f} "
                    f"({meta['elapsed_sec']:.1f}s)",
                    flush=True,
                )

    all_df = (
        pd.DataFrame(results)[SUMMARY_COLUMNS]
        .sort_values(["mds_mode", "pct", "rep"])
        .reset_index(drop=True)
    )
    for mode in modes:
        mode_df = all_df[all_df["mds_mode"] == mode].reset_index(drop=True)
        atomic_write_dataframe(mode_df, args.output_dir / mode / "summary.csv", index=False)

    # Always rebuild summary_all.csv from all mode summary.csv files present on disk
    existing_summaries = [
        pd.read_csv(args.output_dir / m / "summary.csv")
        for m in ("metric", "nonmetric", "ml")
        if (args.output_dir / m / "summary.csv").exists()
    ]
    if existing_summaries:
        combined_df = (
            pd.concat(existing_summaries, ignore_index=True)[SUMMARY_COLUMNS]
            .sort_values(["mds_mode", "pct", "rep"])
            .reset_index(drop=True)
        )
        atomic_write_dataframe(combined_df, args.output_dir / "summary_all.csv", index=False)

    total_wall = time.perf_counter() - t_start
    print(f"\nCompleted {len(tasks)} tasks in {total_wall:.1f}s wall time.")
    print("\nGrouped summary (mean +- SD across replicates):")
    print(format_grouped_summary(all_df))
    print(f"\nOutputs saved under: {args.output_dir}")


if __name__ == "__main__":
    main()
