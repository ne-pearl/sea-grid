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
