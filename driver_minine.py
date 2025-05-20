import pathlib
import random
import numpy as np
from polars import read_csv
from datastructures import Data, Result
from formulations_pyomo import formulate
from plotting import plot

random.seed(0)
np.random.seed(0)
data = pathlib.Path("minine")
tables = dict(
    buses=read_csv(data / "buses.csv"),
    reference_bus="NEMA",
    demands=read_csv(data / "demands.csv"),
    generators=read_csv(data / "generators.csv"),
    lines=read_csv(data / "lines.csv"),
    offers=read_csv(data / "offers.csv").sort(by=["generator_id", "price"]),
)
data = Data.init(**tables)
result: Result = formulate(data)
plot(data=data, result=result, **tables, scale=3.0, k=None)
