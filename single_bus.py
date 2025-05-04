from typing import TypeAlias
import IPython
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model

# Units of measure ===========================================================
Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

# Network data ===============================================================
load = 250.0  # MW
offers = pl.DataFrame(
    [
        {"generator_id": "G1", "max_quantity": 100.0, "price": 10.00},
        {"generator_id": "G2", "max_quantity": 100.0, "price": 11.00},
        {"generator_id": "G1", "max_quantity": 100.0, "price": 12.00},
        {"generator_id": "G2", "max_quantity": 100.0, "price": 13.00},
    ],
    schema={"id": Id, "generator_id": Id, "max_quantity": MW, "price": USDPerMWh},
).sort(by=["generator_id", "price"])


# Unique identifier for each offer
for column in [
    pl.arange(0, pl.len()).over("generator_id").alias("tranche"),
    pl.format("{}/{}", pl.col("generator_id"), pl.col("tranche") + 1).alias("id"),
]:
    offers = offers.with_columns(column)

# Index sets =================================================================
Offers = range(offers.height)


# Model ======================================================================
def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "max_quantity"])


model = pyo.ConcreteModel()
model.p = pyo.Var(Offers, bounds=supply_bounds)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)
model.balance = pyo.Constraint(expr=sum(model.p[o] for o in Offers) == load)
model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
)


# Optimization ===============================================================
def optimize(model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


optimize(model, tee=True)  # tee output to console

# Extract decision variables =================================================
quantity = [pyo.value(model.p[o]) for o in Offers]
marginal_price = model.dual[model.balance]
offers = offers.with_columns(quantity=pl.Series(quantity))
offers = offers.with_columns(  # utilization of each offer
    (pl.col("quantity") / pl.col("max_quantity")).alias("utilization")
)
print("offers -", offers)

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
print("generators -", generators)


def cumsum_mid(x, start=0):
    """Interval midpoints from widths."""
    accumulated = np.concatenate(([start], np.cumsum(x)))
    return (accumulated[:-1] + accumulated[1:]) * 0.5


# Plot dispatch and price
sorted_offers = offers.sort("price")
_, generator_indices = np.unique(sorted_offers[:, "generator_id"], return_inverse=True)
colors = plt.get_cmap("tab10")(generator_indices)
fig, ax = plt.subplots()
bars = ax.bar(
    x=cumsum_mid(sorted_offers["max_quantity"]),
    height=sorted_offers["price"],
    width=sorted_offers["max_quantity"],
    color=colors,
    alpha=0.4,
)
select = sorted_offers["quantity"] > 0
dispatched_offers = sorted_offers.filter(select)
ax.bar(
    x=cumsum_mid(dispatched_offers["quantity"]),
    height=dispatched_offers["price"],
    width=dispatched_offers["quantity"],
    color=colors[select],
    alpha=0.8,
)
ax.axvline(x=load, color="black", label="load")
ax.axhline(y=marginal_price, color="red", label="marginal price")
ax.text(load, 0.1, "load", rotation=90, va="bottom", ha="center", color="black")
ax.text(0.0, marginal_price, "marginal price", va="center", ha="left", color="red")

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
ax.set_xlabel("Quantity (MW)")
ax.set_ylabel("Price ($/MWh)")
ax.set_title("Offer stack to marginal stack")
plt.savefig("single_bus.png", dpi=300)
plt.show(block=False)
