"""Fit 8D Maximum-Likelihood MDS (Nosofsky et al., 2018) to 360-rock GPT-4o similarity matrices.

Fits the Gaussian ML-MDS objective s_hat_ij = u - d_ij (minimizing sum_{i<j} ((u - d_ij) - s_ij)^2
via L-BFGS-B with exact analytic gradients, best of `n_init` random restarts) to both:
  - output/baseline_360/chatgpt_360_similarity_matrix.csv
  - output/encourage_middle_360/chatgpt_encourage_middle_360_similarity_matrix.csv

Rotates each 8D solution orthogonally onto the 6 z-scored human dimension ratings in
data/360_rocks_table.csv (preserving all pairwise Euclidean distances) and saves to:
  - data/MDS Solutions/ml_baseline_8d_rotated.txt
  - data/MDS Solutions/ml_encourage_8d_rotated.txt

Usage:
    python fit_vlm_ml_mds.py [--n-init 10] [--max-iter 1500] [--seed 42]
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
        Path("output/baseline_360/chatgpt_360_similarity_matrix.csv"),
        Path("data/MDS Solutions/ml_baseline_8d_rotated.txt"),
        Path("data/MDS Solutions/rotated_baseline_8d_onto_norm7.txt"),
    ),
    (
        "GPT-4o encourage_middle",
        Path("output/encourage_middle_360/chatgpt_encourage_middle_360_similarity_matrix.csv"),
        Path("data/MDS Solutions/ml_encourage_8d_rotated.txt"),
        Path("data/MDS Solutions/rotated_encourage_8d_onto_norm7.txt"),
    ),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-init", type=int, default=10, help="Number of random L-BFGS-B restarts (default: 10)")
    parser.add_argument("--max-iter", type=int, default=1500, help="Maximum L-BFGS-B iterations per restart (default: 1500)")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (default: 42)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _, ratings_df, target_8d = shm.load_rock_table(shm.ROCKS_TABLE_PATH)
    ratings_arr = ratings_df.to_numpy(dtype=float)
    pub_human = np.loadtxt(shm.PUBLISHED_HUMAN_MDS_PATH)
    iu = np.triu_indices(shm.N_ROCKS, 1)

    for idx, (name, sim_csv, out_txt, old_txt) in enumerate(MODELS):
        S = pd.read_csv(sim_csv, header=None).to_numpy(dtype=float)
        if S.shape != (shm.N_ROCKS, shm.N_ROCKS):
            raise ValueError(f"Expected {(shm.N_ROCKS, shm.N_ROCKS)} in {sim_csv}, got {S.shape}")
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

        s_obs = S[iu]
        d_new = euclidean_distances(rotated)[iu]
        slope, intercept = np.polyfit(-d_new, s_obs, 1)
        pred = intercept - slope * d_new
        pct_var = 100.0 * (1.0 - np.sum((s_obs - pred) ** 2) / np.sum((s_obs - s_obs.mean()) ** 2))
        r_dims = [float(np.corrcoef(rotated[:, d], ratings_arr[:, d])[0, 1]) for d in range(6)]
        r_pub = float(np.corrcoef(pdist(rotated), pdist(pub_human))[0, 1])
        r_old = float(np.corrcoef(pdist(rotated), pdist(np.loadtxt(old_txt)))[0, 1]) if old_txt.exists() else float("nan")

        print(
            f"{name:24s} | saved -> {out_txt} | {elapsed:.1f}s (best_init={res['best_init']}, iter={res['n_iter']}) | "
            f"%Var(sim)={pct_var:.2f}% | r_old={r_old:.4f} | r_pub_human={r_pub:.4f} | mean_r6={np.mean(r_dims):.4f}"
        )


if __name__ == "__main__":
    main()
