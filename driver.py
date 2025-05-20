import random
import numpy as np
import polars as pl
from datastructures import Data, Result
from formulations import formulate, marginal_price

# Set random seed for reproducibility of networkx graph layouts
random.seed(0)
np.random.seed(0)

# System data
tables = dict(
    buses=pl.DataFrame(
        {
            "id": ["B1", "B2", "B3"],
        },
    ),
    reference_bus="B1",
    demands=pl.DataFrame(
        {
            "id": ["D1", "D2"],
            "bus_id": ["B3", "B3"],
            "load": [100.0, 50.0],
        }
    ),
    generators=pl.DataFrame(
        {
            "id": ["G1", "G2", "G3"],
            "bus_id": ["B1", "B2", "B1"],
        }
    ),
    lines=pl.DataFrame(
        {
            "from_bus_id": ["B1", "B1", "B2"],
            "to_bus_id": ["B2", "B3", "B3"],
            "capacity": [30.0, 100.0, 100.0],
            "susceptance": [1000.0, 1000.0, 1000.0],
        }
    ),
    offers=pl.DataFrame(
        {
            "generator_id": ["G1", "G2", "G3", "G1", "G2", "G3"],
            "max_quantity": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
            "price": [10.00, 12.00, 14.00, 20.00, 22.00, 24.00],
        },
    ).sort(by=["generator_id", "price"]),
)

data = Data.init(**tables)
result: Result = formulate(data)
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
