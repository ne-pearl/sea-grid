import argparse
import pathlib
from matplotlib import pyplot as plt
from sea_opf import dcopf, postprocess
from sea_plotting import plot_tables
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
    fig, ax = plot_tables(buses, generators, lines, offers)
    plt.show(block=False)
