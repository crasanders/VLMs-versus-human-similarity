"""Fit 8D Maximum-Likelihood MDS (Nosofsky et al., 2018) to 360-rock VLM similarity data.

Fits the Gaussian ML-MDS objective s_hat_ij = u - d_ij (minimizing sum_{i<j} ((u - d_ij) - s_ij)^2
via L-BFGS-B with exact analytic gradients, best of `n_init` random restarts) to:
  - GPT-4o baseline         (output/baseline_360/chatgpt_360_similarity_matrix.csv)
  - GPT-4o encourage_middle (output/encourage_middle_360/chatgpt_encourage_middle_360_similarity_matrix.csv)
  - Llama-4-Scout-109B/17B  (output/vlm_360/llama4_scout_360/similarities.csv)
  - Qwen2.5-VL-72B          (output/vlm_360/qwen2_5vl_72b_360/similarities.csv)
  - Gemma-3-27B             (output/vlm_360/gemma3_27b_360/similarities.csv)

Rotates each 8D solution orthogonally onto the 6 z-scored human dimension ratings in
data/360_rocks_table.csv (preserving all pairwise Euclidean distances) and saves to
`data/MDS Solutions/ml_<slug>_8d_rotated.txt`.

Usage:
    python fit_vlm_ml_mds.py [--n-init 10] [--max-iter 10000] [--seed 42] [--only-missing]
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from sklearn.metrics import euclidean_distances

import subsample_human_mds as shm

MODELS = [
    (
        "GPT-4o baseline",
        "matrix",
        Path("output/baseline_360/chatgpt_360_similarity_matrix.csv"),
        Path("data/MDS Solutions/ml_baseline_8d_rotated.txt"),
        Path("data/MDS Solutions/rotated_baseline_8d_onto_norm7.txt"),
    ),
    (
        "GPT-4o encourage_middle",
        "matrix",
        Path("output/encourage_middle_360/chatgpt_encourage_middle_360_similarity_matrix.csv"),
        Path("data/MDS Solutions/ml_encourage_8d_rotated.txt"),
        Path("data/MDS Solutions/rotated_encourage_8d_onto_norm7.txt"),
    ),
    (
        "Llama-4-Scout-109B/17B",
        "pairs",
        Path("output/vlm_360/llama4_scout_360/similarities.csv"),
        Path("data/MDS Solutions/ml_llama4_scout_8d_rotated.txt"),
        None,
    ),
    (
        "Qwen2.5-VL-72B",
        "pairs",
        Path("output/vlm_360/qwen2_5vl_72b_360/similarities.csv"),
        Path("data/MDS Solutions/ml_qwen25vl_72b_8d_rotated.txt"),
        None,
    ),
    (
        "Gemma-3-27B",
        "pairs",
        Path("output/vlm_360/gemma3_27b_360/similarities.csv"),
        Path("data/MDS Solutions/ml_gemma3_27b_8d_rotated.txt"),
        None,
    ),
]


def load_similarity_matrix(kind: str, path: Path, rock_ids: list[str]) -> np.ndarray:
    """Load a symmetric 360x360 similarity matrix aligned to canonical rock_ids."""
    n = len(rock_ids)
    if kind == "matrix":
        S = pd.read_csv(path, header=None).to_numpy(dtype=float)
        if S.shape != (n, n):
            raise ValueError(f"Expected {(n, n)} in {path}, got {S.shape}")
        return S

    if kind == "pairs":
        df = pd.read_csv(path)
        idx_map = {rid: i for i, rid in enumerate(rock_ids)}
        r1 = df["Rock1"].map(idx_map)
        r2 = df["Rock2"].map(idx_map)
        if r1.isna().any() or r2.isna().any():
            missing = set(df.loc[r1.isna(), "Rock1"]).union(set(df.loc[r2.isna(), "Rock2"]))
            raise ValueError(f"Unmapped rock IDs in {path}: {sorted(missing)[:5]}")
        rating_col = "Model Rating" if "Model Rating" in df.columns else "ChatGPT Rating"
        vals = df[rating_col].to_numpy(dtype=float)
        S = np.full((n, n), np.nan, dtype=float)
        np.fill_diagonal(S, 9.0)
        i_arr = r1.to_numpy(dtype=int)
        j_arr = r2.to_numpy(dtype=int)
        S[i_arr, j_arr] = vals
        S[j_arr, i_arr] = vals
        return S

    raise ValueError(f"Unknown similarity format kind={kind}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-init", type=int, default=10, help="Number of random L-BFGS-B restarts (default: 10)")
    parser.add_argument("--max-iter", type=int, default=10000, help="Maximum L-BFGS-B iterations per restart (default: 10000)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    parser.add_argument("--only-missing", action="store_true", help="Skip models whose output .txt file already exists")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rock_ids, ratings_df, target_8d = shm.load_rock_table(shm.ROCKS_TABLE_PATH)
    ratings_arr = ratings_df.to_numpy(dtype=float)
    pub_human = np.loadtxt(shm.PUBLISHED_HUMAN_MDS_PATH)
    iu = np.triu_indices(shm.N_ROCKS, 1)

    for idx, (name, kind, sim_csv, out_txt, old_txt) in enumerate(MODELS):
        if not sim_csv.exists():
            print(f"{name:24s} | skipped (missing {sim_csv})")
            continue
        if args.only_missing and out_txt.exists():
            print(f"{name:24s} | cached  -> {out_txt}")
            continue

        S = load_similarity_matrix(kind, sim_csv, rock_ids)
        mask = ~np.isnan(S)
        np.fill_diagonal(mask, False)
        cnts = mask.astype(int)
        S_clean = np.where(mask, S, 0.0)

        t0 = time.perf_counter()
        res = shm.fit_ml_mds(
            mean_sim=S_clean,
            cnts=cnts,
            mask=mask,
            n_components=shm.N_DIMENSIONS,
            n_init=args.n_init,
            max_iter=args.max_iter,
            mds_seed=args.seed + idx,
        )
        elapsed = time.perf_counter() - t0
        rotated = shm.rotate_orthogonal_procrustes(res["X"], target_8d)

        out_txt.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(out_txt, rotated, fmt="%.8f")

        valid = mask[iu]
        s_obs = S[iu][valid]
        d_new = euclidean_distances(rotated)[iu][valid]
        slope, intercept = np.polyfit(-d_new, s_obs, 1)
        pred = intercept - slope * d_new
        pct_var = 100.0 * (1.0 - np.sum((s_obs - pred) ** 2) / np.sum((s_obs - s_obs.mean()) ** 2))
        r_dims = [float(np.corrcoef(rotated[:, d], ratings_arr[:, d])[0, 1]) for d in range(6)]
        r_pub = float(np.corrcoef(pdist(rotated), pdist(pub_human))[0, 1])
        r_old = (
            float(np.corrcoef(pdist(rotated), pdist(np.loadtxt(old_txt)))[0, 1])
            if (old_txt is not None and old_txt.exists())
            else float("nan")
        )

        print(
            f"{name:24s} | saved -> {out_txt} | {elapsed:.1f}s (best_init={res['best_init']}, iter={res['n_iter']}) | "
            f"%Var(sim)={pct_var:.2f}% | r_old={r_old:.4f} | r_pub_human={r_pub:.4f} | mean_r6={np.mean(r_dims):.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
