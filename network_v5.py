import math
from typing import Any, Callable, Iterable, TypeAlias
import IPython
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.base.param import ParamData
from pyomo.core.expr.relational_expr import EqualityExpression, InequalityExpression

# Prevent truncation of tall tables
IPython.core.interactiveshell.InteractiveShell.ast_node_interactivity = "all"
pl.Config.set_tbl_rows(-1)  # "unlimited rows"

# Set random seed for reproducibility of networkx graph layouts
np.random.seed(0)

# Units of measure
Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

##############################################################################
# Network data
##############################################################################

buses = pl.DataFrame(
    {
        "id": ["B1", "B2", "B3"],
    },
    schema={"id": Id},
)

demands = pl.DataFrame(
    {
        "id": ["D1", "D2"],
        "bus_id": ["B3", "B3"],
        "load": [100.0, 50.0],
    },
    schema={"id": Id, "bus_id": Id, "load": MW},
)

buses = buses.join(  # bus_id for each offer
    demands.group_by("bus_id").agg(pl.col("load").sum()),
    left_on="id",
    right_on="bus_id",
    how="left",
).with_columns(pl.col("load").fill_null(0.0))

reference_bus = 0

generators = pl.DataFrame(  # excludes "capacity", "cost"!
    {
        "id": ["G1", "G2", "G3"],
        "bus_id": ["B1", "B2", "B1"],
    },
    schema={"id": Id, "bus_id": Id},
)

lines = pl.DataFrame(
    {
        "from_bus_id": ["B1", "B1", "B2"],
        "to_bus_id": ["B2", "B3", "B3"],
        "susceptance": [1000, 1000, 1000],
        "capacity": [30.0, 100.0, 100.0],
    },
    schema={
        "from_bus_id": Id,
        "to_bus_id": Id,
        "susceptance": MWPerRad,
        "capacity": MW,
    },
)

# Ensure edge pairs are consistely ordered to simplify look-ups
pairs = list(zip(lines[:, "from_bus_id"], lines[:, "to_bus_id"]))
lines = lines.with_columns(
    from_bus_id=pl.Series(min(p) for p in pairs),
    to_bus_id=pl.Series(max(p) for p in pairs),
)

offers = pl.DataFrame(
    {
        "generator_id": ["G1", "G2", "G3", "G1", "G2", "G3"],
        "max_quantity": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0],
        "price": [10.00, 12.00, 14.00, 20.00, 22.00, 24.00],
    },
    schema={"generator_id": Id, "max_quantity": MW, "price": USDPerMWh},
).sort(by=["generator_id", "price"])

# Unique identifier for each offer
for column in [
    pl.arange(0, pl.len()).over("generator_id").alias("tranche"),
    pl.format("{}/{}", pl.col("generator_id"), pl.col("tranche") + 1).alias("id"),
]:
    offers = offers.with_columns(column)

##############################################################################
# Index sets
##############################################################################

Buses = range(buses.height)
Demands = range(demands.height)
Generators = range(generators.height)
Lines = range(lines.height)
Offers = range(offers.height)

##############################################################################
# Optimization model parameters
##############################################################################

model = pyo.ConcreteModel()
model.loads = pyo.Param(
    Buses,
    initialize={b: buses[b, "load"] for b in Buses},
    mutable=True,
    doc="load (demand) @ bus [MW]",
)
model.line_capacities = pyo.Param(
    Lines,
    initialize={ell: lines[ell, "capacity"] for ell in Lines},
    mutable=True,
    doc="capacity @ line [MW]",
)

##############################################################################
# Optimization model decision variables & bounds
##############################################################################


def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "max_quantity"])


def flow_bounds(model: Model, ell: int) -> tuple[float, float]:
    capacity = model.line_capacities[ell]
    return (-capacity, +capacity)


model.p = pyo.Var(Offers, bounds=supply_bounds, doc="power injection @ offer [MW]")
model.f = pyo.Var(
    Lines,
    # bounds=flow_bounds,
    doc="power flow @ line [MW]",
)
model.theta = pyo.Var(
    Buses, bounds=(-math.pi, +math.pi), doc="voltage angle @ bus [rad]"
)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

##############################################################################
# Optimization model objective function
##############################################################################

model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
    doc="total operating cost [$/h]",
)

##############################################################################
# Optimization model equality constraints
##############################################################################


def network_incidence(ell: int, b: int) -> int:
    if buses[b, "id"] == lines[ell, "from_bus_id"]:
        return -1
    if buses[b, "id"] == lines[ell, "to_bus_id"]:
        return +1
    else:
        return 0


def supply_incidence(b: int, g: int, o: int) -> bool:
    bus_generator: bool = buses[b, "id"] == generators[g, "bus_id"]
    generator_offer: bool = generators[g, "id"] == offers[o, "generator_id"]
    return bus_generator and generator_offer


def load_incidence(b: int, d: int) -> bool:
    if buses[b, "id"] == demands[d, "bus_id"]:
        return -1
    else:
        return 0


line_bus_incidence = np.array(
    [[network_incidence(ell, b) for b in Buses] for ell in Lines]
)
bus_offer_incidence = np.array(
    [
        [any(supply_incidence(b, g, o) for g in Generators) for o in Offers]
        for b in Buses
    ],
    dtype=float,
)
bus_load_incidence = np.array(
    [[load_incidence(b, d) for d in Demands] for b in Buses], dtype=float
)


def balance_rule(model: Model, b: int) -> EqualityExpression:
    return (
        sum(
            alpha * model.p[o]
            for o in Offers
            if (alpha := bus_offer_incidence[b, o]) != 0
        )
        + sum(
            beta * model.f[ell]
            for ell in Lines
            if (beta := line_bus_incidence[ell, b]) != 0
        )
        == model.loads[b]
    )


def flow_rule(model: Model, ell: int) -> EqualityExpression:
    return (
        sum(
            gamma * model.theta[b] * lines[ell, "susceptance"]
            for b in Buses
            if (gamma := line_bus_incidence[ell, b]) != 0
        )
        == model.f[ell]
    )


model.balance = pyo.Constraint(Buses, rule=balance_rule, doc="power balance @ bus")
model.flow = pyo.Constraint(Lines, rule=flow_rule, doc="power flow @ line")
model.reference_angle = pyo.Constraint(
    expr=model.theta[reference_bus] == 0, doc="reference voltage angle @ bus[0]"
)

##############################################################################
# Optimization model inequality constraints
##############################################################################


def flow_bound_lower_rule(model: Model, ell: int) -> InequalityExpression:
    return -model.line_capacities[ell] <= model.f[ell]


def flow_bound_upper_rule(model: Model, ell: int) -> InequalityExpression:
    return model.f[ell] <= model.line_capacities[ell]


model.flow_bounds_lower = pyo.Constraint(
    Lines, rule=flow_bound_lower_rule, doc="flow lower bound @ line"
)
model.flow_bounds_upper = pyo.Constraint(
    Lines, rule=flow_bound_upper_rule, doc="flow upper bound @ line"
)

##############################################################################
# Solve optimization model
##############################################################################


# Optimization
def optimize(model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


total_cost = optimize(model, tee=True)  # tee output to console

model.dual.pprint()

##############################################################################
# Extract optimal decision values
##############################################################################

quantity = [pyo.value(model.p[o]) for o in Offers]
flow = [pyo.value(model.f[ell]) for ell in Lines]
angle = [pyo.value(model.theta[b]) for b in Buses]
load_marginal_prices = [model.dual[model.balance[b]] for b in Buses]
flow_marginal_prices = np.add(
    [model.dual[model.flow_bounds_lower[ell]] for ell in Lines],
    [model.dual[model.flow_bounds_upper[ell]] for ell in Lines],
)

##############################################################################
# Sanity checks
##############################################################################

free_bus_ids = [b for b in Buses if b != reference_bus]
K = line_bus_incidence[:, free_bus_ids]
KtB = K.T @ np.diag(lines[:, "susceptance"])
SF = np.linalg.solve(KtB @ K, -KtB).T
injections = bus_offer_incidence @ quantity - buses[:, "load"].to_numpy()

assert np.allclose(lines[:, "susceptance"] * (line_bus_incidence @ angle), flow)
assert np.allclose(
    buses[:, "load"] - bus_offer_incidence @ quantity, line_bus_incidence.T @ flow
)
assert np.allclose(
    buses[:, "load"] - bus_offer_incidence @ quantity,
    line_bus_incidence.T @ (lines[:, "susceptance"] * (line_bus_incidence @ angle)),
)
assert np.allclose(SF @ injections[free_bus_ids], flow)

##############################################################################
# Extend data tables with decision variables
##############################################################################

offers = offers.with_columns(quantity=pl.Series(quantity))
lines = lines.with_columns(flow=pl.Series(flow))
buses = buses.with_columns(
    angle_deg=pl.Series(np.rad2deg(angle)),
    price=pl.Series(load_marginal_prices),
    quantity=pl.Series(
        sum(bus_offer_incidence[b, o] * quantity[o] for o in Offers) for b in Buses
    ),
)

lines = lines.with_columns(  # utilization of each line
    (pl.col("flow").abs() / pl.col("capacity")).alias("utilization")
)
offers = offers.join(  # bus_id for each offer
    generators[:, ("id", "bus_id")], left_on="generator_id", right_on="id", how="left"
)
offers = offers.with_columns(  # utilization of each offer
    (pl.col("quantity") / pl.col("max_quantity")).alias("utilization")
)

print("offers -", offers)
print("lines -", lines)
print("buses -", buses)
print("generators -", generators)

load_payment = sum(buses[b, "load"] * load_marginal_prices[b] for b in Buses)

##############################################################################
# Evaluate marginal prices
##############################################################################


def direct_marginal_price(model, parameter: ParamData, delta=1.0) -> float:
    original = parameter.value
    unperturbed = optimize(model)
    parameter.value += delta
    perturbed = optimize(model)
    parameter.value = original  # restore
    return (perturbed - unperturbed) / delta


load_marginal_price_estimates = [
    direct_marginal_price(model, model.loads[b]) for b in Buses
]
flow_marginal_price_estimates = [
    direct_marginal_price(model, model.line_capacities[ell]) for ell in Lines
]

assert np.allclose(load_marginal_price_estimates, load_marginal_prices)
assert np.allclose(flow_marginal_price_estimates, flow_marginal_prices)

##############################################################################
# Helper functions for plotting
##############################################################################


def mapvalues(
    f: Callable[..., str], keys: Iterable[str], *values: Iterable[Any]
) -> dict[str, str]:
    return dict(zip(keys, map(f, *values)))


def hybrid_layout(
    graph: nx.Graph,
    hub_scale: float = 1.0,
    satellite_scale: float = 1.0,
    k: float = 0.4,
    seed: int = 0,
    **kwargs,
) -> dict:
    """
    This hybrid layout seems to produce better outcome than the pure layout provided by networkx.
    """
    # Work-around: spring_layout allows orientation to influence layout in evil ways
    assert not graph.is_directed()
    hubs = [n for n in graph.nodes if graph.degree(n) > 1]
    pos = nx.kamada_kawai_layout(graph.subgraph(hubs), scale=hub_scale)
    for hub in hubs:
        satellites = {*graph.neighbors(hub)}.difference(hubs)
        satpos = nx.circular_layout(
            graph.subgraph(satellites),
            center=pos[hub],
            scale=satellite_scale,
        )
        pos.update(satpos)
    assert len(pos) == graph.number_of_nodes()
    return nx.spring_layout(graph, pos=pos, fixed=hubs, k=k, seed=seed, **kwargs)


##############################################################################
# Network definition for plotting
##############################################################################

network = nx.DiGraph()

# Node defnition
bus_price_labels = mapvalues("${:.2f}/MWh".format, buses["id"], buses["price"])
offer_price_labels = mapvalues(
    "≤{:}MW\n@ ${:}/MWh".format, offers["id"], offers["max_quantity"], offers["price"]
)
demand_labels = mapvalues("{:}MW".format, demands["id"], demands["load"])

load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
load_cmap = cm.coolwarm  # Or plasma, inferno, magma, coolwarm, etc.
bus_colors = [load_cmap(load_norm(load)) for load in buses["load"]]
bus_colors = ["lightblue" for load in buses["load"]]

generator_cmap = plt.get_cmap(name="tab10", lut=generators.height)
generator_colors = {
    gid: mc.to_hex(generator_cmap(i)) for i, gid in enumerate(generators["id"])
}
offer_colors = [generator_colors[gid] for gid in offers["generator_id"]]
demand_colors = ["red" for demand in demands["id"]]

network.add_nodes_from(buses["id"])
network.add_nodes_from(offers["id"])
network.add_nodes_from(demands["id"])
node_labels = bus_price_labels | offer_price_labels | demand_labels

# Edge defnition
flow_labels = mapvalues(
    "{:.0f}MW/\n{:.0f}MW".format,
    zip(lines["from_bus_id"], lines["to_bus_id"]),
    lines["flow"],
    lines["capacity"],
)
supply_labels = mapvalues(
    "{:.0f}MW".format, zip(offers["id"], offers["bus_id"]), offers["quantity"]
)
demand_load_labels = mapvalues(
    "{:.0f}MW".format,
    zip(demands["bus_id"], demands["id"]),
    demands["load"],
)
network.add_edges_from(zip(lines["from_bus_id"], lines["to_bus_id"]))
network.add_edges_from(zip(offers["id"], offers["bus_id"]))
network.add_edges_from(zip(demands["bus_id"], demands["id"]))
edge_utilization = [
    *lines["utilization"],
    *offers["utilization"],
    *([1.0] * len(Demands)),
]
edge_labels = flow_labels | supply_labels | demand_load_labels

# Network layout
scale = 1.5  # 50.0
pos: dict = hybrid_layout(
    network.to_undirected(), hub_scale=scale, satellite_scale=scale, k=scale
)

plt.figure(figsize=(15, 9), dpi=80)
plt.axis("equal")

# Network annotation
font_size = 8
nx.draw_networkx_nodes(
    network, pos, nodelist=buses["id"], node_color=bus_colors, node_shape="s"
)
nx.draw_networkx_nodes(
    network, pos, nodelist=offers["id"], node_color=offer_colors, node_shape="o"
)
nx.draw_networkx_nodes(
    network, pos, nodelist=demands["id"], node_color=demand_colors, node_shape="^"
)
nx.draw_networkx_labels(
    network,
    pos,
    labels={id: id for id in buses["id"]},
    font_size=font_size,
    font_color="black",
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
    pos,
    labels={id: id for id in demands["id"]},
    font_size=font_size,
    font_color="black",
)
nx.draw_networkx_labels(
    network,
    {id: xy + [0, -0.3 * scale] for id, xy in pos.items()},
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

# Display network
plt.title(
    f"      Paid by Load: ${load_payment:.2f}/h\n"
    f"Paid to Generators: ${total_cost:.2f}/h\n"
    f" Congestion Charge: ${load_payment - total_cost:.2f}/h"
)
plt.axis("off")
plt.margins(x=0.2, y=0.2)
plt.savefig("network.png", dpi=600, bbox_inches="tight")
plt.show(block=False)

##############################################################################
# Congestion analysis
##############################################################################

path_rows: list[dict] = []
path_edge_rows: list[dict] = []
undirected_network = network.to_undirected()

for source in buses[:, "id"]:
    for target in buses[:, "id"]:
        if source == target:
            continue
        node_sequences = nx.all_simple_paths(
            undirected_network, source=source, target=target
        )
        edge_sequences = list(map(nx.utils.pairwise, node_sequences))
        for path_count, sequence in enumerate(edge_sequences, start=1):
            path_id = f"{source}-{target}/{path_count}"
            edge_rows: list[dict] = []
            path_resistance = 0.0
            for from_node_id, to_node_id in sequence:
                orientation = 1 if from_node_id < to_node_id else -1
                if orientation == -1:
                    from_node_id, to_node_id = to_node_id, from_node_id
                susceptance, utilization = (
                    lines.filter(
                        (pl.col("from_bus_id") == from_node_id)
                        & (pl.col("to_bus_id") == to_node_id)
                    )
                    .select(["susceptance", "utilization"])
                    .row(0)
                )
                path_resistance += 1 / susceptance
                path_edge_rows.append(
                    dict(
                        from_node_id=from_node_id,
                        to_node_id=to_node_id,
                        orientation=orientation,
                        is_congested=0.99 < utilization,
                        path_id=path_id,
                        # source=source,
                        # target=target,
                    )
                )
            path_rows.append(
                dict(
                    path_id=path_id,
                    source=source,
                    target=target,
                    susceptance=1 / path_resistance,
                )
            )
            # for row in edge_rows:
            #     row["path_susceptance"] = 1 / path_resistance
            # path_edge_rows.extend(edge_rows)

paths = pl.DataFrame(path_rows)
path_edges = pl.DataFrame(path_edge_rows)

group_keys = ["source", "target"]
grouped = paths.group_by(group_keys).agg(
    pl.col("susceptance").sum().alias("total_susceptance")
)
paths = paths.join(grouped, on=group_keys).with_columns(
    (pl.col("susceptance") / pl.col("total_susceptance")).alias("relative_susceptance")
)

path_edges = path_edges.join(
    paths[:, ("path_id", "source", "target", "relative_susceptance")], on=["path_id"]
)

print("paths -", paths)
print("path_edges -", path_edges)

congested_path_edges = path_edges.filter(pl.col("is_congested"))
print("congested path_edges -", congested_path_edges)

input_nodes = set(offers[:, "bus_id"])

path_summary = paths.group_by(["target"]).agg(pl.all()).sort(["target"])
congestion_summary = (
    congested_path_edges.group_by(["from_node_id", "to_node_id", "target"])
    .agg(pl.all())
    .sort(["target"])
)

print("path_summary -", path_summary)
print("congestion_summary -", congestion_summary)

print("Constraints implied by congestion:")
for target, coeffs, sources in congestion_summary.select(
    "target", "relative_susceptance", "source"
).iter_rows():
    terms = [
        f"{c:.2f}*PTotal[{s}]" for c, s in zip(coeffs, sources) if s in input_nodes
    ]
    print(f"@ {target}: {' + '.join(terms)} == 0")

print("Constraints implied unit increment:")
for target, sources in path_summary.select("target", "source").iter_rows():
    terms = [f"PTotal[{s}]" for s in set(sources) if s in input_nodes]
    print(f"@ {target}: {' + '.join(terms)} == 1")
