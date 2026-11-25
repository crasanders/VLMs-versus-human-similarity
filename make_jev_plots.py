"""Plot model dimension and similarity ratings against human ratings for the 30 rocks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

OUTPUT_FOLDER = Path("output") / "jev" / "plots"

DIMENSIONS = ["Lightness of Color", "Grain Size", "Roughness", "Shininess", "Organization", "Saturation"]
# Human Saturation is Munsell chroma (0.3-5.7), not a 1-9 rating
HUMAN_AXIS_LABELS = {"Saturation": "Human (Munsell chroma)"}

SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
HUMAN_COLOR = "#7a7974"
# Dimension plots: one row per model, one column per dimension
DIMENSION_MODELS = {
    "Jev": {"file": Path("output") / "jev" / "dimension_ratings.csv", "column": "Model Rating", "color": "#2a78d6"},
    "Decisions (gpt-6-luna)": {
        "file": Path("output") / "openai_decisions" / "dimension_ratings.csv",
        "column": "Model Rating",
        "color": "#1baf7a",
    },
    "GPT-4o": {"file": Path("chatgpt_ratings.csv"), "column": "ChatGPT Rating", "color": "#eb6834"},
}

# Similarity plots: one column per model, one row per condition
SIMILARITY_MODELS = {
    "Jev": {"color": "#2a78d6", "folder": Path("output") / "jev", "column": "Model Rating"},
    "Decisions (gpt-6-luna)": {"color": "#1baf7a", "folder": Path("output") / "openai_decisions", "column": "Model Rating"},
    "GPT-4o": {"color": "#eb6834", "column": "ChatGPT Rating"},
}
# Conditions to plot, one row each. Add {"No anchors": "no_anchors"} to compare with the unanchored runs.
SIMILARITY_CONDITIONS = {"Anchors": "anchors"}
# GPT-4o runs from get_similarities.py (base prompt), in their own run folders
GPT4O_FILES = {
    "No anchors": Path("output") / "baseline" / "similarities.csv",
    "Anchors": Path("output") / "anchors" / "similarities.csv",
}


def similarity_file(model, condition):
    if model == "GPT-4o":
        return GPT4O_FILES[condition]
    return SIMILARITY_MODELS[model]["folder"] / f"similarities_{SIMILARITY_CONDITIONS[condition]}.csv"


def panel_title(model, condition):
    # Name the condition only when more than one is plotted
    return model if len(SIMILARITY_CONDITIONS) == 1 else f"{model}, {condition.lower()}"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_SECONDARY,
        "axes.titlecolor": TEXT_PRIMARY,
        "axes.titlesize": 11,
        "xtick.color": TEXT_SECONDARY,
        "ytick.color": TEXT_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 9,
    }
)


def load_ratings():
    """One row per rock and dimension, with the human rating and a column per model."""
    ratings = None
    for model, spec in DIMENSION_MODELS.items():
        data = pd.read_csv(spec["file"]).rename(columns={spec["column"]: model})
        if ratings is None:
            ratings = data[["Rock", "Dimension", "Human Rating", model]]
        else:
            ratings = ratings.merge(data[["Rock", "Dimension", model]], on=["Rock", "Dimension"], how="left")
    return ratings


def dimension_correlations(ratings):
    """Per dimension: each model against the humans, and each pair of models against each other."""
    models = list(DIMENSION_MODELS)
    rows = []
    for dim in DIMENSIONS:
        data = ratings[ratings["Dimension"] == dim]
        row = {"Dimension": dim}
        row.update({f"{m} vs Human": data[m].corr(data["Human Rating"]) for m in models})
        row.update(
            {f"{a} vs {b}": data[a].corr(data[b]) for i, a in enumerate(models) for b in models[i + 1 :]}
        )
        rows.append(row)
    table = pd.DataFrame(rows).set_index("Dimension")
    table.loc["Mean"] = table.mean()
    return table


def style_axes(ax):
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_scatter(ratings):
    fig, axes = plt.subplots(
        len(DIMENSION_MODELS), len(DIMENSIONS), figsize=(2.6 * len(DIMENSIONS), 3.3 * len(DIMENSION_MODELS)),
        sharex=True, sharey=True,
    )
    for row, (model, spec) in enumerate(DIMENSION_MODELS.items()):
        for col, dim in enumerate(DIMENSIONS):
            ax = axes[row, col]
            data = ratings[ratings["Dimension"] == dim]
            style_axes(ax)
            ax.plot([0, 9], [0, 9], color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
            ax.scatter(data["Human Rating"], data[model], s=24, color=spec["color"], edgecolor=SURFACE,
                       linewidth=1, zorder=2)
            r = data["Human Rating"].corr(data[model])
            ax.set_title(f"{dim}\n{model.split(' (')[0]}, r = {r:.2f}", loc="left", fontsize=9)
            # Human Saturation (Munsell chroma) goes below 1, so the axes start at 0
            ax.set_xlim(-0.2, 9.5)
            ax.set_ylim(-0.2, 9.5)
            ax.set_xticks(range(0, 10, 2))
            ax.set_yticks(range(0, 10, 2))
            ax.set_aspect("equal")
            if row == len(DIMENSION_MODELS) - 1:
                ax.set_xlabel(HUMAN_AXIS_LABELS.get(dim, "Human rating"))
            if col == 0:
                ax.set_ylabel(f"{model}\nrating")
    fig.suptitle("Model vs. human dimension ratings (30 rocks)", x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_histograms(ratings):
    bins = np.arange(-0.5, 10, 1)
    fig, axes = plt.subplots(
        len(DIMENSION_MODELS), len(DIMENSIONS), figsize=(2.6 * len(DIMENSIONS), 2.3 * len(DIMENSION_MODELS)),
        sharex=True, sharey=True,
    )
    for row, (model, spec) in enumerate(DIMENSION_MODELS.items()):
        for col, dim in enumerate(DIMENSIONS):
            ax = axes[row, col]
            data = ratings[ratings["Dimension"] == dim]
            style_axes(ax)
            ax.grid(axis="x", visible=False)
            ax.hist(data[model], bins=bins, color=spec["color"], edgecolor=SURFACE, linewidth=2,
                    label=model.split(" (")[0])
            # Human ratings as an outline in every panel for reference
            ax.hist(data["Human Rating"], bins=bins, histtype="step", color=HUMAN_COLOR, linewidth=2,
                    label="Human (Munsell chroma)" if dim == "Saturation" else "Human")
            if row == 0:
                ax.set_title(dim, loc="left")
            if col == 0:
                ax.set_ylabel(f"{model}\nnumber of rocks")
            if row == len(DIMENSION_MODELS) - 1:
                ax.set_xlabel("Rating")
            if col == len(DIMENSIONS) - 1 or row == 0 and col == 0:
                ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY, fontsize=8)
    axes[0, 0].set_xticks(range(0, 10, 2))
    axes[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))
    fig.suptitle("Distribution of dimension ratings (30 rocks)", x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def load_similarities():
    """Map (model, condition) to that run's pair ratings, with columns Rock1, Rock2, Human Rating, Rating."""
    similarities = {}
    for model, spec in SIMILARITY_MODELS.items():
        for condition in SIMILARITY_CONDITIONS:
            path = similarity_file(model, condition)
            if path is None:
                continue
            if not path.exists():
                print(f"Skipping {model}, {condition}: {path} not found")
                continue
            data = pd.read_csv(path)[["Rock1", "Rock2", "Human Rating", spec["column"]]]
            similarities[model, condition] = data.rename(columns={spec["column"]: "Rating"})
    return similarities


def similarity_grid(similarities, panel_size):
    """A condition x model grid of axes; cells with no run show a note instead of a plot."""
    fig, axes = plt.subplots(
        len(SIMILARITY_CONDITIONS),
        len(SIMILARITY_MODELS),
        figsize=(panel_size[0] * len(SIMILARITY_MODELS), panel_size[1] * len(SIMILARITY_CONDITIONS)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    cells = []
    for row, condition in enumerate(SIMILARITY_CONDITIONS):
        for col, model in enumerate(SIMILARITY_MODELS):
            ax = axes[row, col]
            if (model, condition) not in similarities:
                ax.set_axis_off()
                ax.text(0.5, 0.5, f"{model}\n{condition.lower()}: not run", ha="center", va="center",
                        color=TEXT_SECONDARY, transform=ax.transAxes)
                continue
            style_axes(ax)
            cells.append((ax, model, condition, similarities[model, condition]))
    return fig, axes, cells


def plot_similarity_scatter(similarities):
    fig, axes, cells = similarity_grid(similarities, (3.8, 4.8))
    for ax, model, condition, data in cells:
        ax.plot([1, 9], [1, 9], color=TEXT_SECONDARY, linestyle="--", linewidth=1, zorder=1)
        ax.scatter(
            data["Human Rating"],
            data["Rating"],
            s=16,
            color=SIMILARITY_MODELS[model]["color"],
            alpha=0.6,
            edgecolor="none",
            zorder=2,
        )
        r = data["Human Rating"].corr(data["Rating"])
        ax.set_title(f"{panel_title(model, condition)}\nr = {r:.2f}", loc="left")
        ax.set_xlim(0.5, 9.5)
        ax.set_ylim(0.5, 9.5)
        ax.set_xticks(range(1, 10))
        ax.set_yticks(range(1, 10))
        ax.set_aspect("equal")
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("Human rating")
        ax.set_ylabel("Model rating")
    fig.suptitle("Model vs. human similarity ratings (435 pairs of 30 rocks)", x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def plot_similarity_histograms(similarities):
    bins = np.arange(0.5, 10, 1)
    fig, axes, cells = similarity_grid(similarities, (3.8, 2.6))
    for ax, model, condition, data in cells:
        ax.grid(axis="x", visible=False)
        ax.hist(data["Rating"], bins=bins, color=SIMILARITY_MODELS[model]["color"], edgecolor=SURFACE,
                linewidth=2, label=model.split(" (")[0])
        # Human ratings as an outline in every panel for reference
        ax.hist(data["Human Rating"], bins=bins, histtype="step", color=HUMAN_COLOR, linewidth=2, label="Human")
        ax.set_title(
            f"{panel_title(model, condition)}\nmean = {data['Rating'].mean():.2f} (human {data['Human Rating'].mean():.2f})",
            loc="left",
        )
        ax.set_xticks(range(1, 10))
        ax.tick_params(labelbottom=True, labelleft=True)
        ax.set_xlabel("Similarity rating")
        ax.set_ylabel("Pairs")
        ax.legend(loc="upper right", frameon=False, labelcolor=TEXT_SECONDARY)
    fig.suptitle("Distribution of similarity ratings", x=0.01, ha="left", color=TEXT_PRIMARY)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def save(fig, name):
    for ext in ["pdf", "png"]:
        fig.savefig(OUTPUT_FOLDER / f"{name}.{ext}", dpi=150, bbox_inches="tight")


def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    ratings = load_ratings()
    correlations = dimension_correlations(ratings)
    correlations.round(3).to_csv(OUTPUT_FOLDER / "dimension_correlations.csv")
    print(correlations.round(2).to_string())
    save(plot_scatter(ratings), "dimension_ratings_scatter")
    save(plot_histograms(ratings), "dimension_ratings_histograms")
    similarities = load_similarities()
    save(plot_similarity_scatter(similarities), "similarity_ratings_scatter")
    save(plot_similarity_histograms(similarities), "similarity_ratings_histograms")
    print(f"Saved plots to {OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
