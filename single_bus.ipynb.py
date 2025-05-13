# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# Module imports

# %%
from typing import TypeAlias
import IPython
from IPython.display import display
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model

# %% [markdown]
# Units of measure

# %%
Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

# %% [markdown]
# System configuration: Just one node

# %%
load = 250.0  # MW
offers = pl.DataFrame(
    {
        "generator_id": ["G1", "G2", "G1", "G2"],
        "max_quantity": [100.0, 100.0, 100.0, 100.0],
        "price": [10.00, 11.00, 12.00, 13.00],
    },
    schema={"generator_id": Id, "max_quantity": MW, "price": USDPerMWh},
).sort(by=["generator_id", "price"])

# %% [markdown]
# Add a unique identifier for each offer

# %%
for column in [
    pl.arange(0, pl.len()).over("generator_id").alias("tranche"),
    pl.format("{}/{}", pl.col("generator_id"), pl.col("tranche") + 1).alias("id"),
]:
    offers = offers.with_columns(column)

# %% [markdown]
# ![](./single_bus_plot.png)

# %% [markdown]
# Index set

# %%
Offers = range(offers.height)

# %% [markdown]
# Optimization model in PyOMO

# %%
model = pyo.ConcreteModel()

# %% [markdown]
# Associate load parameter

# %%
model.load_ = pyo.Param(initialize=load, mutable=True, within=pyo.Reals)  # "model.load" is reserved


# %% [markdown]
# Decision variables

# %%
def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "max_quantity"])


model.p = pyo.Var(Offers, bounds=supply_bounds)

# %% [markdown]
# Power balance constraint

# %%
model.balance = pyo.Constraint(expr=sum(model.p[o] for o in Offers) == model.load_)

# %% [markdown]
# Dual variable for marginal price

# %%
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# %% [markdown]
# Optimization objective function

# %%
model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
)


# %% [markdown]
# Solve the optimization problem

# %%
def optimize(model, tee=False) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, tee=tee)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


optimize(model, tee=True)  # tee output to console

# %% [markdown]
# Extract optimal values of decision variables

# %%
quantity = [pyo.value(model.p[o]) for o in Offers]
offers = offers.with_columns(quantity=pl.Series(quantity))
offers = offers.with_columns(  # utilization of each offer
    (pl.col("quantity") / pl.col("max_quantity")).alias("utilization")
)
generators = (
    offers.group_by("generator_id")
    .agg(
        [
            pl.col("max_quantity").sum(),
            pl.col("quantity").sum(),
        ]
    )
    .with_columns([(pl.col("quantity") / pl.col("max_quantity")).alias("utilization")])
)

# %% [markdown]
# Extract marginal price

# %%
marginal_price = model.dual[model.balance]

# %% [markdown]
# Verify marginal price by direct calculation

# %%
cost_unperturbed = optimize(model)
model.load_[None] += 1.0
cost_perturbed = optimize(model)
marginal_price_estimate = (cost_perturbed - cost_unperturbed) / 1.0
assert abs(marginal_price_estimate - marginal_price) < 1e-6


# %% [markdown]
# Helper function for placing bars in the bar chart

# %%
def cumsum_mid(x, start=0):
    """Interval midpoints from widths."""
    accumulated = np.concatenate(([start], np.cumsum(x)))
    return (accumulated[:-1] + accumulated[1:]) * 0.5


# %% [markdown]
# Initialize plotting axes

# %% [markdown]
# Values required for plot

# %%
fig, ax = plt.subplots()
ax.set_xlabel("Quantity (MW)")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("Offer stack to marginal stack")
plt.close(fig)

# %%
sorted_offers = offers.sort("price")
select = sorted_offers["quantity"] > 0
dispatched_offers = sorted_offers.filter(select)
_, generator_indices = np.unique(sorted_offers[:, "generator_id"], return_inverse=True)
colors = plt.get_cmap("tab10")(generator_indices)

# %% [markdown]
# Plot bars and annotations

# %%
bars = ax.bar(
    x=cumsum_mid(sorted_offers["max_quantity"]),
    height=sorted_offers["price"],
    width=sorted_offers["max_quantity"],
    color=colors,
    alpha=0.4,
)
ax.bar(
    x=cumsum_mid(dispatched_offers["quantity"]),
    height=dispatched_offers["price"],
    width=dispatched_offers["quantity"],
    color=colors[select],
    alpha=0.8,
)

ax.bar_label(
    bars,
    labels=sorted_offers[:, "id"],
    label_type="center",
    padding=5,
)
ax.bar_label(
    bars,
    labels=sorted_offers.select([pl.format("{}MW", "max_quantity")]).to_series(),
    label_type="center",
    padding=-5,
)
ax.bar_label(
    bars,
    labels=sorted_offers.select(
        pl.col("price").map_elements("${:.2f}/MWh".format, return_dtype=str)
    ).to_series(),
    label_type="edge",
)

# %% [markdown]
# Add lines and annotations

# %%
ax.axvline(x=load, color="black", label="load")
ax.axhline(y=marginal_price, color="red", label="marginal price")
ax.text(load, 0.1, "load", rotation=90, va="bottom", ha="center", color="black")
ax.text(0.0, marginal_price, "marginal price", va="center", ha="left", color="red")

# %% [markdown]
# Display plot

# %%
plt.savefig("single_bus.png", dpi=300)
display(fig)
