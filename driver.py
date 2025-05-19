import polars as pl
from datastructures import Data, Result
from formulations import formulate


##############################################################################
# Data data
##############################################################################

buses = pl.DataFrame(
    {
        "id": ["B1", "B2", "B3"],
    },
)

demands = pl.DataFrame(
    {
        "id": ["D1", "D2"],
        "bus_id": ["B3", "B3"],
        "load": [100.0, 50.0],
    }
)

generators = pl.DataFrame(  # excludes "capacity", "cost"!
    {
        "id": ["G1", "G2", "G3"],
        "bus_id": ["B1", "B2", "B1"],
    }
)

lines = pl.DataFrame(
    {
        "from_bus_id": ["B1", "B1", "B2"],
        "to_bus_id": ["B2", "B3", "B3"],
        "capacity": [30.0, 100.0, 100.0],
        "susceptance": [1000., 1000., 1000.],
    }
)

offers = pl.DataFrame(
    {
        "generator_id": ["G1", "G2", "G3", "G1", "G2", "G3"],
        "max_quantity": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
        "price": [10.00, 12.00, 14.00, 20.00, 22.00, 24.00],
    },
).sort(by=["generator_id", "price"])


net = Data.init(
    buses=buses,
    demands=demands,
    generators=generators,
    lines=lines,
    offers=offers,
)
result: Result = formulate(net)
