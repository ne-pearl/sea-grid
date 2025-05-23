import argparse
import pathlib
import numpy as np
from matplotlib import pyplot as plt
from sea_opf import dcopf, postprocess
from sea_plotting import make_network, plot_network, plot_opf
from pjm5bus_pandas import buses, generators, lines, offers, reference_bus

if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Run SCED")
    # parser.add_argument("directory", type=pathlib.Path)
    # parser.add_argument("--font_size", type=int, default=10)
    # parser.add_argument("--scale_x", type=float, default=1.0)
    # parser.add_argument("--scale_y", type=float, default=1.0)
    # args = parser.parse_args()
    #
    # directory: pathlib.Path = args.directory
    # assert directory.exists(), f'"{directory.resolve()}" does not exist'

    total_cost = dcopf(buses, generators, lines, offers, reference_bus)
    postprocess(buses, generators, lines, offers)

    # Create blank figure and plotting axes
    fig, ax = plt.subplots(figsize=np.array([15, 9]) * 0.9, dpi=150, frameon=False)
    ax.axis("equal")

    network, pos = make_network(buses=buses, lines=lines, offers=offers, scale_x=2.0)
    plot_network(
        fig=fig,
        ax=ax,
        network=network,
        pos=pos,
        buses=buses,
        generators=generators,
        offers=offers,
    )
    plot_opf(
        fig=fig,
        ax=ax,
        network=network,
        pos=pos,
        buses=buses,
        lines=lines,
        offers=offers,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show(block=False)
