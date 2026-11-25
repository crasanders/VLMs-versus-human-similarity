import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from pathlib import Path
import numpy as np
import os
import argparse


def plot_images(
    img_dir,
    x,
    y,
    xlabel,
    ylabel,
    img_scale=0.2,
    fig_size=36,
    file_format=".png",
    correlate=False,
    save_file=None,
):
    fig = plt.figure(figsize=(fig_size, fig_size))
    ax = fig.add_subplot(1, 1, 1)
    sorted_files = sorted(Path(img_dir).rglob("*.png"))
    for i, file in enumerate(sorted_files):
        image = plt.imread(file)
        im = OffsetImage(image, zoom=img_scale)
        x0 = x[i]
        y0 = y[i]
        artists = []
        ab = AnnotationBbox(im, (x0, y0), xycoords="data", frameon=False)
        artists.append(ax.add_artist(ab))
        ax.update_datalim(np.column_stack([x0, y0]))
        ax.autoscale()
    for spine in ax.spines.values():
        spine.set_visible(False)
    if correlate:
        plt.plot(plt.ylim(), plt.ylim(), color="black", linestyle="--", linewidth=4)
        corr = np.corrcoef(x, y)[0, 1]
        ax.annotate("r = {}".format(round(corr, 2)), xy=(max(x), min(y)), size=50)
    plt.xlabel(xlabel, fontsize=fig_size)
    plt.ylabel(ylabel, fontsize=fig_size)
    plt.tick_params(labelsize=fig_size / 2)
    if save_file is not None:
        plt.savefig(save_file)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot MDS correlation plots.")
    parser.add_argument(
        "--file",
        "-f",
        type=str,
        required=True,
        help="Filename of the MDS solution (e.g., rotated_encourage_8d_onto_norm7.txt)",
    )
    parser.add_argument(
        "--index",
        "-i",
        type=int,
        required=True,
        help="Dimension index to plot (0-based)",
    )
    args = parser.parse_args()

    rocks_path = "data\\Rocks Library\\360 Rocks"
    figs_path = "output\\mds correlation plots\\"

    human_mds = np.loadtxt("rotated_mds_rocks360.txt")
    mds_solutions_path = Path("data/MDS Solutions")

    f = Path(args.file)
    mds = np.loadtxt(mds_solutions_path / f)
    os.makedirs(figs_path + f.stem, exist_ok=True)

    i = args.index  # Use command line argument
    plot_images(
        rocks_path,
        mds[:, i],
        human_mds[:, i],
        f"ChatGPT Dim {i + 1}",
        f"Human Dim {i + 1}",
        img_scale=0.1,
        save_file=f"{figs_path + f.stem}/{f.stem}-{i + 1}.pdf",
    )
