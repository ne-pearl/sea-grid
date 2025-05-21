import argparse
import pathlib
from matplotlib import pyplot as plt
from datastructures import Data, Result, load
from formulations import formulate
from plotting import display

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SCED")
    parser.add_argument("directory", type=pathlib.Path)
    parser.add_argument("--font_size", type=int, default=10)
    parser.add_argument("--scale_x", type=float, default=1.0)
    parser.add_argument("--scale_y", type=float, default=1.0)
    args = parser.parse_args()

    directory: pathlib.Path = args.directory
    assert directory.exists(), f'"{directory.resolve()}" does not exist'
    tables = load(directory)
    data = Data.init(**tables)
    result: Result = formulate(data)
    display(
        data=data,
        result=result,
        **tables,
        font_size=args.font_size,
        scale_x=args.scale_x,
        scale_y=args.scale_y,
    )

    # ====================================================================
    import numpy as np

    bus_indices = range(data.bus_load.size)
    free_bus_ids = [b for b in bus_indices if b != data.reference_bus]
    K = data.line_bus_incidence[:, free_bus_ids]
    KtB = K.T @ np.diag(1.0 / data.line_reactance)
    SF = np.linalg.solve(KtB @ K, -KtB).T
    injections = result.dispatch_quantity @ data.offer_bus_incidence - data.bus_load
    assert np.allclose(injections, -result.line_flow @ data.line_bus_incidence)
    assert np.allclose(SF @ injections[free_bus_ids], result.line_flow)
    assert np.allclose(
        (data.line_bus_incidence @ result.voltage_angle)
        * data.base_power
        / data.line_reactance,
        result.line_flow,
    )

