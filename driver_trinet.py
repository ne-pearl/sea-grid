from matplotlib import pyplot as plt
from datastructures import Data, Result, load
from formulations_pyomo import formulate
from plotting import display

tables = load("trinet")
data = Data.from_dataframes(**tables)
result: Result = formulate(data)
display(data=data, result=result, **tables, scale_x=2.0)
plt.show(block=False)