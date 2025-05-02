import math
from typing import TypeAlias
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.expr.relational_expr import EqualityExpression

Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

# Network data ===============================================================
# load in MW
buses = pl.DataFrame(
    [
        {"id": "B1", "load": 0.0},
        {"id": "B2", "load": 0.0},
        {"id": "B3", "load": 150.0},
    ],
    schema={"id": Id, "load": MW},
)
generators = pl.DataFrame(
    [  # excludes "capacity", "cost"
        {"id": "G1", "node_id": "B1"},
        {"id": "G2", "node_id": "B2"},
        {"id": "G3", "node_id": "B1"},
    ],
    schema={"id": Id, "node_id": Id},
)

# capacity in MW
# susceptance in MW/rad
lines = pl.DataFrame(
    # fmt: off
    [
        {"from_node_id": "B1", "to_node_id": "B2", "susceptance": 1000, "capacity": 30.0},
        {"from_node_id": "B1", "to_node_id": "B3", "susceptance": 1000, "capacity": 100.0},
        {"from_node_id": "B2", "to_node_id": "B3", "susceptance": 1000, "capacity": 100.0},
    ],
    schema={"from_node_id": Id, "to_node_id": Id, "susceptance": MWPerRad, "capacity": MW},
    # fmt: on
)
# price in $/MWh
# max_quantity in MW
offers = pl.DataFrame(
    [
        {"id": "G1a", "generator_id": "G1", "max_quantity": 200.0, "price": 10.00},
        {"id": "G2a", "generator_id": "G2", "max_quantity": 200.0, "price": 12.00},
        {"id": "G3a", "generator_id": "G3", "max_quantity": 200.0, "price": 14.00},
        {"id": "G1b", "generator_id": "G1", "max_quantity": 100.0, "price": 20.00},
        {"id": "G2b", "generator_id": "G2", "max_quantity": 100.0, "price": 22.00},
        {"id": "G3b", "generator_id": "G3", "max_quantity": 100.0, "price": 24.00},
    ],
    schema={"id": Id, "generator_id": Id, "max_quantity": MW, "price": USDPerMWh},
).sort(by=["generator_id", "price"])

# Index sets =================================================================
Buses = range(buses.height)
Lines = range(lines.height)
Offers = range(offers.height)
Generators = range(generators.height)

# Model parameters ===========================================================
model = pyo.ConcreteModel()


model.loads = pyo.Param(
    Buses, initialize={b: buses[b, "load"] for b in Buses}, mutable=True
)
model.susceptances = pyo.Param(
    Lines, initialize={ell: lines[ell, "susceptance"] for ell in Lines}
)


# Model variables ============================================================
def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "max_quantity"])


def flow_bounds(model: Model, ell: int) -> tuple[float, float]:
    capacity = lines[ell, "capacity"]
    return (-capacity, +capacity)


model.p = pyo.Var(Offers, bounds=supply_bounds)
model.f = pyo.Var(Lines, bounds=flow_bounds)
model.theta = pyo.Var(Buses, bounds=(-math.pi, +math.pi))
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Model objective function ===================================================
model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
)

# Model constraints ==========================================================


def network_incidence(b: int, ell: int) -> int:
    if buses[b, "id"] == lines[ell, "from_node_id"]:
        return -1
    if buses[b, "id"] == lines[ell, "to_node_id"]:
        return +1
    else:
        return 0


def supply_incidence(b: int, g: int, o: int) -> bool:
    node_generator: bool = buses[b, "id"] == generators[g, "node_id"]
    generator_offer: bool = generators[g, "id"] == offers[o, "generator_id"]
    return node_generator and generator_offer


def balance_rule(model: Model, b: int) -> EqualityExpression:
    return (
        sum(model.p[o] for g in Generators for o in Offers if supply_incidence(b, g, o))
        + sum(
            sign * model.f[ell] for ell in Lines if (sign := network_incidence(b, ell))
        )
        == model.loads[b]
    )


def flow_rule(model: Model, ell: int) -> EqualityExpression:
    return (
        sum(
            sign * model.susceptances[b] * model.theta[b]
            for b in Buses
            if (sign := network_incidence(b, ell))
        )
        == model.f[ell]
    )


model.balance = pyo.Constraint(Buses, rule=balance_rule)
model.flow = pyo.Constraint(Lines, rule=flow_rule)
model.reference_angle = pyo.Constraint(expr=model.theta[0] == 0.0)

# Optimization ===============================================================
solver = pyo.SolverFactory("highs")
results = solver.solve(model, tee=True)  # tee output to console
assert results.solver.termination_condition is pyo.TerminationCondition.optimal
print(results)

# Transfer solution to tables ================================================
supply = [pyo.value(model.p[o]) for o in Offers]
flow = [pyo.value(model.f[ell]) for ell in Lines]
angles_deg = [pyo.value(model.theta[b]) * 180 / math.pi for b in Buses]
marginal_prices = [model.dual[model.balance[b]] for b in Buses]

offers = offers.with_columns([pl.Series("supply", supply)])
lines = lines.with_columns([pl.Series("flow", flow)])
buses = buses.with_columns(
    [pl.Series("angle_deg", angles_deg), pl.Series("price", marginal_prices)]
)

# Added aggregated offer data into generators table
for column in [
    offers.group_by("generator_id").agg(pl.col("price").alias("prices_list")),
    offers.group_by("generator_id").agg(pl.col("supply").sum()),
    offers.group_by("generator_id").agg(pl.col("max_quantity").sum()),
]:
    generators = generators.join(
        column, left_on="id", right_on="generator_id", how="left"
    )

# Form price label string from list of offer prices
generators = generators.with_columns(
    pl.col("prices_list")
    .map_elements(lambda x: "/".join(map("{:.2f}".format, sorted(x))))
    .alias("prices")
)

print("offers -", offers)
print("lines -", lines)
print("buses -", buses)
print("generators -", generators)


# Evaluate marginal prices ===================================================


def marginal_price_estimate(model, b: int, delta_load: float = 1.0) -> float:

    # Restore nominal state before perturbation
    # (copy.deepcopy of pyomo models is not officially supported)
    results = solver.solve(model, tee=False)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    f_model = pyo.value(model.total_cost)
    load_original = model.loads[b]

    # Perturb model with unit increment load at bus b
    model.loads[b] += delta_load
    results = solver.solve(model, tee=True)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    f_perturbed = pyo.value(model.total_cost)

    # Restore model state
    model.loads[b] = load_original

    # Difference approximation of marginal cost
    return (f_perturbed - f_model) / delta_load


marginal_price_estimates = [marginal_price_estimate(model, b) for b in Buses]
print(f"marginal_price_estimates = {marginal_price_estimates}")

# Helper functions for plotting ==============================================


def mapvalues(f, keys, *values) -> dict:
    return dict(zip(keys, map(f, *values)))


def offset(positions: dict, dx: float, dy: float) -> dict:
    dxy = np.array([dx, dy])
    return {id: xy + dxy for id, xy in positions.items()}


# Network definition for plotting =============================================
network = nx.DiGraph()

network.add_nodes_from(buses["id"])
network.add_nodes_from(generators["id"])
bus_color = ["lightblue" for _ in buses["id"]]
generator_color = ["lightgreen" for _ in generators["id"]]
node_color = bus_color + generator_color

network.add_edges_from(zip(lines["from_node_id"], lines["to_node_id"]))
network.add_edges_from(zip(generators["id"], generators["node_id"]))
# line_color = tags(lines["from_node_id"], "black")
# connection_color = tags(generators["id"], "gray")
# edge_color = line_color + connection_color

nodal_price_labels = mapvalues(
    "{}MW\n@ ${:.2f}/MWh".format, buses["id"], buses["load"], buses["price"]
)
offer_prices_labels = mapvalues(
    "$({:})/MWh".format, generators["id"], generators["prices"]
)
flow_labels = mapvalues(
    "{:.0f}MW/{:.0f}MW".format,
    zip(lines["from_node_id"], lines["to_node_id"]),
    lines["flow"],
    lines["capacity"],
)
supply_labels = mapvalues(
    "{:.0f}MW/{:.0f}MW".format,
    zip(generators["id"], generators["node_id"]),
    generators["supply"],
    generators["max_quantity"],
)
line_utilization = lines["flow"].abs() / lines["capacity"]
connection_utilization = [0.0 for _ in generators["id"]]
edge_utilization = [*line_utilization, *connection_utilization]

options = dict(
    with_labels=True,
    node_color=node_color,
    edge_color=edge_utilization,
    edge_cmap=cm.cool,
    edge_vmin=0.0,
    edge_vmax=1.0,
)

pos = nx.spring_layout(network, k=3 / math.sqrt(network.number_of_nodes()), seed=0)
nx.draw_networkx(network, pos, **options)
nx.draw_networkx_labels(
    network,
    offset(pos, 0, -0.1),
    labels=nodal_price_labels | offer_prices_labels,
    font_size=6,
)
nx.draw_networkx_edge_labels(
    network,
    pos,
    edge_labels=flow_labels | supply_labels,
    font_size=6,
)

plt.axis("off")
plt.tight_layout()
plt.savefig("network.png", dpi=300)
plt.show()
