"""Predict Category 1 classification probabilities P(C1) for the 34 schematic faces using Jev.

Provides the 10 training exemplars from Experiment 1A of Nosofsky (1991) in `state`
(Faces 1-5 in Category 1, Faces 6-10 in Category 2) and asks Jev's Choice primitive
to classify each of the 34 faces into Category 1 or Category 2.

Usage:
    python fast_model_face_classification.py [--test] [--faces-per-request N]
"""

import argparse
import asyncio
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typesafe_sdk import AsyncTypeSafeClient, Choice, Image

from fast_model_face_mds import (
    CAT1_COLOR,
    CAT1_EXEMPLARS,
    CAT2_COLOR,
    CAT2_EXEMPLARS,
    JEV_COLOR,
    load_face_thumbnail,
    place_non_overlapping_labels,
)
from make_jev_plots import GRID, SURFACE, TEXT_PRIMARY, TEXT_SECONDARY, style_axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

MODEL = "jev-latest"
IMAGE_HEADERS = {"TypeSafe-Danger-Zone-Images": "true"}
MAX_CONCURRENT_REQUESTS = 8

IMAGE_FOLDER = Path("data") / "34 Faces"
TABLE_1_FILE = Path("data") / "nosofsky1991_table_1.csv"
JEV_OUTPUT_FOLDER = Path("output") / "jev"
PLOTS_OUTPUT_FOLDER = Path("output") / "jev" / "plots"

# State holds 10 training exemplar images; 6 query faces per request = 16 images total (under the 17-image limit)
DEFAULT_FACES_PER_REQUEST = 6
DEFAULT_TEST_N = 4

CLASSIFICATION_QUESTION = "Which category does this face belong to?"
CATEGORIES = {"Category 1": None, "Category 2": None}

_images: dict[str, Image] = {}


def load_image(face: str) -> Image:
    """Reuse one Image per face so each file is read only once."""
    if face not in _images:
        _images[face] = Image(IMAGE_FOLDER / f"{face}.png")
    return _images[face]


def training_state() -> list:
    """Build the state containing the 5 Category 1 exemplars (1-5) followed by the 5 Category 2 exemplars (6-10)."""
    state = []
    for f_num in sorted(CAT1_EXEMPLARS):
        state.extend([
            "This is an example of a face in Category 1.",
            load_image(f"face_{f_num:02d}"),
        ])
    for f_num in sorted(CAT2_EXEMPLARS):
        state.extend([
            "This is an example of a face in Category 2.",
            load_image(f"face_{f_num:02d}"),
        ])
    return state


async def classify_faces_jev(client, semaphore, faces: list[int]):
    state = training_state()
    questions = {
        f"face_{f_num:02d}": Choice(
            instructions=[CLASSIFICATION_QUESTION, load_image(f"face_{f_num:02d}")],
            criteria=CATEGORIES,
        )
        for f_num in faces
    }
    async with semaphore:
        response = await client.system_one(
            state=state,
            questions=questions,
            extra_headers=IMAGE_HEADERS,
        )
    print(f"Classified {len(faces)} faces ({response.usage.input_tokens} input tokens)")
    results = []
    for f_num in faces:
        face_name = f"face_{f_num:02d}"
        answer = response.answers[face_name]
        p_c1 = answer.probabilities.get("Category 1", 0.0)
        p_c2 = answer.probabilities.get("Category 2", 0.0)
        results.append(
            {
                "Face": f_num,
                "Face Name": face_name,
                "Choice": answer.choice,
                "Confidence": answer.confidence,
                "Jev_PC1": p_c1,
                "Jev_PC2": p_c2,
                "Model": response.model,
            }
        )
    return results, response.usage.input_tokens or 0


def save_fig(fig: plt.Figure, path: Path) -> None:
    for ext in ["png", "pdf"]:
        fig.savefig(path.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_classification_scatter(df: pd.DataFrame, path: Path, use_thumbnails: bool = False) -> None:
    """Compare Jev P(C1) against Human Observed P(C1) (All subjects and Learners only) as in Figure 5 of Nosofsky (1991)."""
    comparisons = [
        ("Exp1A_All_Obs_PC1", "Human observed P(C₁) — All subjects"),
        ("Exp1A_Learners_Obs_PC1", "Human observed P(C₁) — Learners only"),
        ("Exp1A_All_Pred_PC1", "Nosofsky (1991) exemplar model P(C₁) — All subjects"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), sharey=True)
    lim = (-0.05, 1.05)
    ticks = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    for ax_idx, (ax, (col, xlabel)) in enumerate(zip(axes, comparisons)):
        style_axes(ax)
        x = df[col].to_numpy(dtype=float)
        y = df["Jev_PC1"].to_numpy(dtype=float)
        r = np.corrcoef(x, y)[0, 1]
        sse = float(np.sum((x - y) ** 2))

        ax.plot([0, 1], [0, 1], color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)

        for _, row in df.iterrows():
            f_num = int(row["Face"])
            face_name = row["Face Name"]
            x0, y0 = float(row[col]), float(row["Jev_PC1"])
            is_cat1 = f_num in CAT1_EXEMPLARS
            is_cat2 = f_num in CAT2_EXEMPLARS
            color = CAT1_COLOR if is_cat1 else (CAT2_COLOR if is_cat2 else JEV_COLOR)

            if use_thumbnails:
                image = load_face_thumbnail(face_name)
                z_base = 4 if (is_cat1 or is_cat2) else 2
                ax.add_artist(AnnotationBbox(OffsetImage(image, zoom=0.38), (x0, y0), frameon=False, zorder=z_base))
                if is_cat1:
                    ax.scatter([x0], [y0], marker="o", s=550, facecolors="none", edgecolors=CAT1_COLOR, linewidths=1.4, zorder=z_base + 1)
                elif is_cat2:
                    ax.scatter([x0], [y0 - 0.008], marker="^", s=750, facecolors="none", edgecolors=CAT2_COLOR, linewidths=1.4, zorder=z_base + 1)
            else:
                marker = "o" if is_cat1 else ("^" if is_cat2 else "o")
                size = 55 if (is_cat1 or is_cat2) else 32
                ax.scatter([x0], [y0], marker=marker, s=size, color=color, edgecolor=SURFACE, linewidth=0.8, zorder=3)

        faces_list = df["Face"].astype(int).tolist()
        labels = [str(f) for f in faces_list]
        colors = [
            CAT1_COLOR if f in CAT1_EXEMPLARS else (CAT2_COLOR if f in CAT2_EXEMPLARS else TEXT_PRIMARY)
            for f in faces_list
        ]
        weights = ["bold" if (f in CAT1_EXEMPLARS or f in CAT2_EXEMPLARS) else "normal" for f in faces_list]
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
            box_w_norm=0.048,
            box_h_norm=0.036,
            zorder=10,
        )

        if ax_idx == 0 and not use_thumbnails:
            ax.scatter([], [], marker="o", s=45, color=CAT1_COLOR, label="Category 1 training (1–5)")
            ax.scatter([], [], marker="^", s=50, color=CAT2_COLOR, label="Category 2 training (6–10)")
            ax.scatter([], [], marker="o", s=32, color=JEV_COLOR, label="Transfer stimuli (11–34)")
            ax.legend(loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID, fontsize=8)

        ax.set_title(f"{xlabel.split(' — ')[1]}\nr = {r:.3f}, SSE = {sse:.3f}", loc="left", fontsize=10)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_aspect("equal")
        ax.set_xlabel(xlabel)
        if ax_idx == 0:
            ax.set_ylabel("Jev predicted P(C₁)")

    fig.suptitle(
        "Jev predicted vs. human & exemplar-model Category 1 probabilities P(C₁) (34 faces, Nosofsky 1991 Table 1)",
        x=0.01,
        ha="left",
        color=TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save_fig(fig, path)


async def run_classification(args):
    table1 = pd.read_csv(TABLE_1_FILE)
    all_faces = table1["Face"].astype(int).tolist()
    faces = all_faces[: args.test_n] if args.test else all_faces

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    faces_per_req = args.faces_per_request
    async with AsyncTypeSafeClient(model=MODEL, timeout=15.0) as client:
        batches = await asyncio.gather(
            *(
                classify_faces_jev(client, semaphore, faces[i : i + faces_per_req])
                for i in range(0, len(faces), faces_per_req)
            )
        )

    results = pd.DataFrame([row for rows, _ in batches for row in rows])
    merged = results.merge(table1, on="Face", how="left")

    output_folder = JEV_OUTPUT_FOLDER / "test" if args.test else JEV_OUTPUT_FOLDER
    output_folder.mkdir(parents=True, exist_ok=True)
    out_csv = output_folder / "face_classifications.csv"
    merged.to_csv(out_csv, index=False)

    total_tokens = sum(t for _, t in batches)
    print(f"\nSaved {len(merged)} face classification predictions to {out_csv} ({total_tokens} input tokens total)")

    if not args.test:
        PLOTS_OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
        plot_classification_scatter(merged, PLOTS_OUTPUT_FOLDER / "face_classification_scatter", use_thumbnails=False)
        plot_classification_scatter(merged, PLOTS_OUTPUT_FOLDER / "face_classification_scatter_faces", use_thumbnails=True)

        for col, label in [
            ("Exp1A_All_Obs_PC1", "Exp 1A All Subjects (Observed)"),
            ("Exp1A_Learners_Obs_PC1", "Exp 1A Learners Only (Observed)"),
            ("Exp1A_All_Pred_PC1", "Exp 1A All Subjects (Exemplar Model Predicted)"),
            ("Exp1A_Learners_Pred_PC1", "Exp 1A Learners Only (Exemplar Model Predicted)"),
        ]:
            r = merged["Jev_PC1"].corr(merged[col])
            r2 = r ** 2
            sse = float(np.sum((merged["Jev_PC1"] - merged[col]) ** 2))
            print(f"  vs. {label}: r = {r:.3f} (% Var = {100 * r2:.1f}%), SSE = {sse:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="store_true", help="Only classify the first --test-n faces")
    parser.add_argument("--test-n", type=int, default=DEFAULT_TEST_N)
    parser.add_argument("--faces-per-request", type=int, default=DEFAULT_FACES_PER_REQUEST)
    asyncio.run(run_classification(parser.parse_args()))


if __name__ == "__main__":
    main()
