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
        ("Bus1", 0.0, 0.0, 1.0),
        ("Bus2", 0.0, 2.0, 1.0),
        ("Bus3", 150.0, 1.0, 0.0),
    ],
    schema={"id": Id, "load": MW, "x": Distance, "y": Distance},
    orient="row",
)
reference_bus = "Bus1"

# %%
generators = DataFrame(
    [
        ("G1", "Bus1"),
        ("G2", "Bus2"),
    ],
    schema={"id": Id, "bus_id": Id},
    orient="row",
)

# %%
lines = DataFrame(
    [
        ("Bus1", "Bus2", 150.0, 0.25),
        ("Bus1", "Bus3", 150.0, 0.25),
        ("Bus2", "Bus3", 150.0, 0.25),
    ],
    schema={"from_bus_id": Id, "to_bus_id": Id, "capacity": MW, "reactance": PU},
    orient="row",
)

# %%
offers = DataFrame(
    [
        ("G1", 200.0, 10.0),
        ("G2", 200.0, 12.0),
    ],
    schema={"generator_id": Id, "quantity": MW, "price": USDPerMWh},
    orient="row",
)
