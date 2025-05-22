# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %% [markdown]
# External dependencies

# %%
from polars import DataFrame
from sea_schema import *

# %% [markdown]
# System data

# %%
buses = DataFrame(
    [
        ("A", 0.0, 0.0, 0.0),
        ("B", 300.0, 1.0, 0.0),
        ("C", 300.0, 2.0, 0.0),
        ("D", 400.0, 2.0, 1.0),
        ("E", 0.0, 0.0, 2.0),
    ],
    schema={"id": Id, "load": MW, "x": Distance, "y": Distance},
    orient="row",
)
reference_bus = "A"

# %%
generators = DataFrame(
    [
        ("Alta", "A"),
        ("ParkCity", "A"),
        ("Solitude", "C"),
        ("Sundance", "D"),
        ("Brighton", "E"),
    ],
    schema={"id": Id, "bus_id": Id},
    orient="row",
)

# %%
lines = DataFrame(
    [
        ("A", "B", 400.0, 2.81),
        ("A", "D", 1000.0, 3.04),
        ("A", "E", 1000.0, 0.64),
        ("B", "C", 1000.0, 1.08),
        ("C", "D", 1000.0, 2.97),
        ("D", "E", 240.0, 2.97),
    ],
    schema={"from_bus_id": Id, "to_bus_id": Id, "capacity": MW, "reactance": PU},
    orient="row",
)

# %%
offers = DataFrame(
    [
        ("Alta", 40.0, 14.0),
        ("ParkCity", 170.0, 15.0),
        ("Solitude", 520.0, 30.0),
        ("Sundance", 200.0, 40.0),
        ("Brighton", 600.0, 10.0),
    ],
    schema={"generator_id": Id, "quantity": MW, "price": USDPerMWh},
    orient="row",
)
