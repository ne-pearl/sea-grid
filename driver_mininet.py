import pathlib
import random
from matplotlib import pyplot as plt
import numpy as np
from polars import read_csv
from datastructures import Data, Result
from formulations_pyomo import formulate
from plotting import plot

random.seed(101)
np.random.seed(101)
basedir = pathlib.Path("mininet")
tables = dict(
    buses=read_csv(basedir / "buses.csv"),
    reference_bus="A",
    generators=read_csv(basedir / "generators.csv"),
    lines=read_csv(basedir / "lines.csv"),
    offers=read_csv(basedir / "offers.csv").sort(by=["generator_id", "price"]),
)
basedir = Data.init(**tables)
result: Result = formulate(basedir)
plot(data=basedir, result=result, **tables, scale_x=2.0)
plt.show(block=False)
