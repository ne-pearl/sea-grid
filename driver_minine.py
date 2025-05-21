import pathlib
import random
from matplotlib import pyplot as plt
import numpy as np
from polars import read_csv
from datastructures import Data, Result
from formulations_pyomo import formulate
from plotting import plot

random.seed(0)
np.random.seed(0)
basedir = pathlib.Path("minine")
tables = dict(
    buses=read_csv(basedir / "buses.csv"),
    reference_bus="NEMA",
    generators=read_csv(basedir / "generators.csv"),
    lines=read_csv(basedir / "lines.csv"),
    offers=read_csv(basedir / "offers.csv").sort(by=["generator_id", "price"]),
)
basedir = Data.init(**tables)
result: Result = formulate(basedir)
plot(data=basedir, result=result, **tables, kscale=2)
plt.show(block=False)