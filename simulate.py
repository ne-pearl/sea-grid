import random
from matplotlib import pyplot as plt
import numpy as np
from polars import DataFrame
from datastructures import Data, Result
from formulations import formulate, marginal_price
from plotting import display

# Set random seed for reproducibility of networkx graph layouts
random.seed(0)
np.random.seed(0)

# System data
tables = dict(
    buses=DataFrame(
        {
            "id": ["B1", "B2", "B3"],
            "load": [0.0, 0.0, 150.0],
            "x": [0., 2., 1.],
            "y": [1., 1., 0.],
        },
    ),
    reference_bus="B1",
    generators=DataFrame(
        {
            "id": ["G1", "G2", "G3"],
            "bus_id": ["B1", "B2", "B1"],
        }
    ),
    lines=DataFrame(
        {
            "from_bus_id": ["B1", "B1", "B2"],
            "to_bus_id": ["B2", "B3", "B3"],
            "capacity": [30.0, 100.0, 100.0],
            "reactance": [1000.0, 1000.0, 1000.0],
        }
    ),
    offers=DataFrame(
        {
            "generator_id": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "quantity": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
            "price": [10.00, 12.00, 14.00, 20.00, 22.00, 24.00],
        },
    ).sort(by=["generator_id", "price"]),
)

data = Data.from_dataframes(**tables)
result: Result = formulate(data)
display(data=data, result=result, **tables, font_size=12, scale_x = 1.5)
plt.show(block=False)

model = result.model
assert np.isclose(
    result.total_cost,
    np.dot(result.dispatch_quantity @ data.offer_bus_incidence, result.energy_price),
)

# fmt: off
energy_price = [
    marginal_price(model, parameter)
    for parameter in model.bus_loads.values()
]
congestion_price = [
    marginal_price(model, parameter)
    for parameter in model.line_capacities.values()
]
# fmt: on
assert np.allclose(energy_price, result.energy_price)
assert np.allclose(congestion_price, result.congestion_price)
