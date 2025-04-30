import math
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.expr.relational_expr import EqualityExpression
from polars import DataFrame, Series, col

# Base power
base_power = 1000

# Network data
nodes = DataFrame(
    [
        {"id": "N1", "load": 0},
        {"id": "N2", "load": 0},
        {"id": "N3", "load": 150},
    ]
)
generators = DataFrame(
    [  # excludes "capacity", "cost"
        {"id": "G1", "node_id": "N1"},
        {"id": "G2", "node_id": "N2"},
        {"id": "G3", "node_id": "N1"},
    ]
)
lines = DataFrame(
    # fmt: off
    [
        {"from_node_id": "N1", "to_node_id": "N2", "susceptance": 0.25, "capacity": 30.},
        {"from_node_id": "N1", "to_node_id": "N3", "susceptance": 0.25, "capacity": 300.},
        {"from_node_id": "N2", "to_node_id": "N3", "susceptance": 0.25, "capacity": 300.},
    ]
    # fmt: on
)
offers = DataFrame(
    [
        {"id": "G1a", "generator_id": "G1", "max_quantity": 200.0, "price": 10.00},
        {"id": "G2a", "generator_id": "G2", "max_quantity": 200.0, "price": 12.00},
        {"id": "G3a", "generator_id": "G3", "max_quantity": 200.0, "price": 14.00},
        # {"id": "G1b", "generator_id": "G1", "max_quantity": 100.0, "price": 20.00},
        # {"id": "G2b", "generator_id": "G2", "max_quantity": 100.0, "price": 22.00},
        # {"id": "G3b", "generator_id": "G3", "max_quantity": 100.0, "price": 24.00},
    ]
).sort(by=["generator_id", "price"])

# Index sets
Nodes = range(nodes.height)
Lines = range(lines.height)
Offers = range(offers.height)
Generators = range(generators.height)

model = pyo.ConcreteModel()


def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "max_quantity"])


def flow_bounds(model: Model, ell: int) -> tuple[float, float]:
    capacity = lines[ell, "capacity"]
    return (-capacity, +capacity)


# Variables
model.p = pyo.Var(Offers, bounds=supply_bounds)
model.f = pyo.Var(Lines, bounds=flow_bounds)
model.theta = pyo.Var(Nodes, bounds=(-math.pi, +math.pi))
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Objective
model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
)


def network_incidence(n: int, ell: int) -> int:
    if nodes[n, "id"] == lines[ell, "from_node_id"]:
        return -1
    if nodes[n, "id"] == lines[ell, "to_node_id"]:
        return +1
    else:
        return 0


def supply_incidence(n: int, g: int, o: int) -> bool:
    node_generator: bool = nodes[n, "id"] == generators[g, "node_id"]
    generator_offer: bool = generators[g, "id"] == offers[o, "generator_id"]
    return node_generator and generator_offer


def balance_rule(model: Model, n: int) -> EqualityExpression:
    return (
        sum(model.p[o] for g in Generators for o in Offers if supply_incidence(n, g, o))
        + sum(network_incidence(n, ell) * model.f[ell] for ell in Lines)
        == nodes[n, "load"]
    )


def flow_rule(model: Model, ell: int) -> EqualityExpression:
    return (
        sum(
            base_power
            * sign
            * lines[ell, "susceptance"]
            * model.theta[n]
            for n in Nodes
            if (sign := network_incidence(n, ell))
        )
        == model.f[ell]
    )


model.balance = pyo.Constraint(Nodes, rule=balance_rule)
model.flow = pyo.Constraint(Lines, rule=flow_rule)
model.reference_angle = pyo.Constraint(expr=model.theta[0] == 0.0)

solver = pyo.SolverFactory("highs")
results = solver.solve(model, tee=True)  # tee output to console

supply = [pyo.value(model.p[o]) for o in Offers]
flow = [pyo.value(model.f[ell]) for ell in Lines]
angles_deg = [pyo.value(model.theta[n]) * 180 / math.pi for n in Nodes]
prices = [model.dual[model.balance[n]] for n in Nodes]

offers = offers.with_columns([Series("supply", supply)])
lines = lines.with_columns([Series("flow", flow)])
nodes = nodes.with_columns([Series("theta_deg", angles_deg), Series("price", prices)])

supply_prices = offers.group_by("generator_id").agg(col("price"))
supply_totals = offers.group_by("generator_id").agg(col("supply").sum())
generators = generators.join(
    supply_totals, left_on="id", right_on="generator_id", how="left"
)
generators = generators.join(
    supply_prices, left_on="id", right_on="generator_id", how="left"
)
generators = generators.with_columns(
    col("price")
    .map_elements(lambda x: ", ".join(map("{:.2f}".format, sorted(x))))
    .alias("prices")
)

print("offers -", offers)
print("lines -", lines)
print("nodes -", nodes)
print("generators -", generators)


def mapvalues(f, keys, *values) -> dict:
    return dict(zip(keys, map(f, *values)))


def offset(pos: dict, dx: float, dy: float) -> dict:
    dxy = np.array([dx, dy])
    return {node: xy + dxy for node, xy in pos.items()}


network = nx.DiGraph()

network.add_nodes_from(nodes["id"])
network.add_nodes_from(generators["id"])
bus_color = ["lightblue" for _ in nodes["id"]]
generator_color = ["lightgreen" for _ in generators["id"]]
node_color = bus_color + generator_color

network.add_edges_from(zip(lines["from_node_id"], lines["to_node_id"]))
network.add_edges_from(zip(generators["id"], generators["node_id"]))
# line_color = tags(lines["from_node_id"], "black")
# connection_color = tags(generators["id"], "gray")
# edge_color = line_color + connection_color

nodal_price_labels = mapvalues("${:.2f}/MWh".format, nodes["id"], nodes["price"])
offer_price_labels = mapvalues(
    "${:}/MWh".format, generators["id"], generators["prices"]
)
flow_labels = mapvalues(
    "{:.0f}MW/{:.0f}MW".format,
    zip(lines["from_node_id"], lines["to_node_id"]),
    lines["flow"],
    lines["capacity"],
)
supply_labels = mapvalues(
    "{:.0f}MW".format,
    zip(generators["id"], generators["node_id"]),
    generators["supply"],
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
    labels=nodal_price_labels | offer_price_labels,
    font_size=9,
)
nx.draw_networkx_edge_labels(
    network, pos, edge_labels=flow_labels | supply_labels, font_size=8
)

plt.axis("off")
plt.tight_layout()
plt.show()
