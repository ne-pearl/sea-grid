import itertools
import math
from typing import TypeAlias
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.expr.relational_expr import EqualityExpression

# Units of measure ===========================================================
Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

# Network data ===============================================================
buses = pl.DataFrame(
    [
        {"id": "B1", "load": 0.0},
        {"id": "B2", "load": 0.0},
        {"id": "B3", "load": 150.0},
    ],
    schema={"id": Id, "load": MW},
)
generators = pl.DataFrame(
    [  # excludes "capacity", "cost"!
        {"id": "G1", "node_id": "B1"},
        {"id": "G2", "node_id": "B2"},
        {"id": "G3", "node_id": "B1"},
    ],
    schema={"id": Id, "node_id": Id},
)

lines = pl.DataFrame(
    # fmt: off
    [
        {"from_bus_id": "B1", "to_bus_id": "B2", "susceptance": 1000, "capacity": 30.0},
        {"from_bus_id": "B1", "to_bus_id": "B3", "susceptance": 1000, "capacity": 100.0},
        {"from_bus_id": "B2", "to_bus_id": "B3", "susceptance": 1000, "capacity": 100.0},
    ],
    schema={"from_bus_id": Id, "to_bus_id": Id, "susceptance": MWPerRad, "capacity": MW},
    # fmt: on
)
offers = pl.DataFrame(
    [
        {"generator_id": "G1", "max_quantity": 200.0, "price": 10.00},
        {"generator_id": "G2", "max_quantity": 200.0, "price": 12.00},
        {"generator_id": "G3", "max_quantity": 200.0, "price": 14.00},
        {"generator_id": "G1", "max_quantity": 100.0, "price": 20.00},
        {"generator_id": "G2", "max_quantity": 100.0, "price": 22.00},
        {"generator_id": "G3", "max_quantity": 100.0, "price": 24.00},
        {"generator_id": "G1", "max_quantity": 200.0, "price": 15.00},
        {"generator_id": "G2", "max_quantity": 200.0, "price": 16.00},
        {"generator_id": "G3", "max_quantity": 200.0, "price": 17.00},
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


# Model decision variables ===================================================
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
    if buses[b, "id"] == lines[ell, "from_bus_id"]:
        return -1
    if buses[b, "id"] == lines[ell, "to_bus_id"]:
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
            sign * model.f[ell]
            for ell in Lines
            if (sign := network_incidence(b, ell)) != 0
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
def optimize(model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


optimize(model, tee=True)  # tee output to console

# Transfer solution to tables ================================================
supply = [pyo.value(model.p[o]) for o in Offers]
flow = [pyo.value(model.f[ell]) for ell in Lines]
angles_deg = [pyo.value(model.theta[b]) * 180 / math.pi for b in Buses]
marginal_prices = [model.dual[model.balance[b]] for b in Buses]

# Extend data tables with decision variables
offers = offers.with_columns([pl.Series("supply", supply)])
lines = lines.with_columns([pl.Series("flow", flow)])
buses = buses.with_columns(
    [
        pl.Series("angle_deg", angles_deg),
        pl.Series("price", marginal_prices),
    ]
)

# Extend data tables with post-processed quantities
lines = lines.with_columns(  # utilization of each line
    (pl.col("flow").abs() / pl.col("capacity")).alias("utilization")
)
offers = offers.join(  # node_id for each offer
    generators[:, ("id", "node_id")], left_on="generator_id", right_on="id", how="left"
)
offers = offers.with_columns(  # utilization of each offer
    (pl.col("supply") / pl.col("max_quantity")).alias("utilization")
)

print("offers -", offers)
print("lines -", lines)
print("buses -", buses)
print("generators -", generators)

# Evaluate marginal prices ===================================================


def marginal_price_estimate(model, b: int, delta_load: float = 1.0) -> float:
    # Restore nominal state before perturbation
    # (copy.deepcopy of pyomo models is not officially supported)
    f_unperturbed = optimize(model)
    load_original = model.loads[b]
    model.loads[b] += delta_load  # increment load at bus b
    f_perturbed = optimize(model)
    model.loads[b] = load_original  # restore model for next time
    return (f_perturbed - f_unperturbed) / delta_load


marginal_price_estimates = [marginal_price_estimate(model, b) for b in Buses]
print(f"marginal_price_estimates = {marginal_price_estimates}")
assert marginal_price_estimates == marginal_prices

# Helper functions for plotting ==============================================


def mapvalues(f, keys, *values) -> dict:
    return dict(zip(keys, map(f, *values)))


def hybrid_layout(
    G: nx.Graph, scale: float = 1.0, k: float = 0.4, seed: int = 0
) -> dict:
    """Produces a better outcome than either pure layout provided by networkx"""
    hubs = [n for n in G.nodes if G.degree(n) > 1]
    pos = nx.kamada_kawai_layout(G.subgraph(hubs), scale=scale)
    for hub in hubs:
        satellites = {
            *network.successors(hub),
            *network.predecessors(hub),
        }.difference(hubs)
        pos.update(nx.circular_layout(G.subgraph(satellites), center=pos[hub]))
        print(f"{hub} => {satellites}")
    assert len(pos) == G.number_of_nodes()
    return nx.spring_layout(G, pos=pos, fixed=hubs, seed=seed, k=k)


# Network definition for plotting =============================================
network = nx.DiGraph()

# Node defnition ==============================================================
bus_price_labels = mapvalues(
    "{:}MW\n@ ${:.2f}/MWh".format, buses["id"], buses["load"], buses["price"]
)
offer_price_labels = mapvalues(
    "{:}MW\n@ ${:}/MWh".format, offers["id"], offers["max_quantity"], offers["price"]
)
load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
load_cmap = cm.coolwarm  # Or plasma, inferno, magma, coolwarm, etc.
bus_colors = [load_cmap(load_norm(load)) for load in buses["load"]]
bus_colors = ["lightblue" for load in buses["load"]]

generator_cmap = plt.get_cmap(name="tab10", lut=generators.height)
generator_colors = {
    gid: mc.to_hex(generator_cmap(i)) for i, gid in enumerate(generators["id"])
}
offer_colors = [generator_colors[gid] for gid in offers["generator_id"]]

network.add_nodes_from(buses["id"])
network.add_nodes_from(offers["id"])
node_labels = bus_price_labels | offer_price_labels

# Edge defnition =============================================================
flow_labels = mapvalues(
    "{:.0f}MW/{:.0f}MW".format,
    zip(lines["from_bus_id"], lines["to_bus_id"]),
    lines["flow"],
    lines["capacity"],
)
supply_labels = mapvalues(
    "{:.0f}MW/{:.0f}MW".format,
    zip(offers["id"], offers["node_id"]),
    offers["supply"],
    offers["max_quantity"],
)
network.add_edges_from(zip(lines["from_bus_id"], lines["to_bus_id"]))
network.add_edges_from(zip(offers["id"], offers["node_id"]))
edge_utilization = [*lines["utilization"], *offers["utilization"]]
edge_labels = flow_labels | supply_labels

# Network layout =============================================================
scale = 20.0
pos: dict = hybrid_layout(network, scale=scale, k=scale * 1.0)

# Network annotation =========================================================
font_size = 8
nx.draw_networkx_nodes(
    network, pos, nodelist=buses["id"], node_color=bus_colors, node_shape="s"
)
nx.draw_networkx_nodes(
    network, pos, nodelist=offers["id"], node_color=offer_colors, node_shape="o"
)
nx.draw_networkx_labels(
    network,
    pos,
    labels={id: id for id in buses["id"]},
    font_size=font_size,
    font_color="red",
)
nx.draw_networkx_labels(
    network,
    pos,
    labels={id: id for id in offers["id"]},
    font_size=font_size,
    font_color="black",
)
nx.draw_networkx_labels(
    network,
    {id: xy + [0, -0.2 * scale] for id, xy in pos.items()},
    labels=node_labels,
    font_size=font_size,
)
nx.draw_networkx_edges(
    network,
    pos,
    edge_color=edge_utilization,
    edge_cmap=cm.coolwarm,
    edge_vmin=0.0,
    edge_vmax=1.0,
)
nx.draw_networkx_edge_labels(
    network,
    pos,
    edge_labels=edge_labels,
    font_size=font_size,
    font_color="blue",
)

# Display network ============================================================
plt.axis("off")
plt.tight_layout()
plt.savefig("network.png", dpi=300)
plt.show(block=True)
