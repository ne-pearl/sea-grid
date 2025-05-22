import math
from typing import TypeAlias
import IPython
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.expr.relational_expr import EqualityExpression

# Prevent truncation of tall tables
IPython.core.interactiveshell.InteractiveShell.ast_node_interactivity = "all"
pl.Config.set_tbl_rows(-1)  # "unlimited rows"

# Units of measure
Id: TypeAlias = pl.String
MW: TypeAlias = pl.Float64
MWPerRad: TypeAlias = pl.Float64
USDPerMWh: TypeAlias = pl.Float64

# Network data
buses = pl.DataFrame(
    {
        "id": ["B1", "B2", "B3"],
        "load": [0.0, 0.0, 150.0],
    },
    schema={"id": Id, "load": MW},
)
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
        "reactance": [1000, 1000, 1000],
        "capacity": [30.0, 100.0, 100.0],
    },
    schema={
        "from_bus_id": Id,
        "to_bus_id": Id,
        "reactance": MWPerRad,
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
        "generator_id": ["G1", "G2", "G3", "G1", "G2", "G3", "G1", "G2", "G3"],
        "quantity": [200.0, 200.0, 200.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
        "price": [10.00, 12.00, 14.00, 20.00, 22.00, 24.00, 15.00, 16.00, 17.00],
    },
    schema={"generator_id": Id, "quantity": MW, "price": USDPerMWh},
).sort(by=["generator_id", "price"])

# Unique identifier for each offer
for column in [
    pl.arange(0, pl.len()).over("generator_id").alias("tranche"),
    pl.format("{}/{}", pl.col("generator_id"), pl.col("tranche") + 1).alias("id"),
]:
    offers = offers.with_columns(column)

# Index sets
Buses = range(buses.height)
Lines = range(lines.height)
Offers = range(offers.height)
Generators = range(generators.height)

# Model parameters
model = pyo.ConcreteModel()
model.loads = pyo.Param(
    Buses,
    initialize={b: buses[b, "load"] for b in Buses},
    mutable=True,
    doc="load (demand) @ bus [MW]",
)
model.reactances = pyo.Param(
    Lines,
    initialize={ell: lines[ell, "reactance"] for ell in Lines},
    doc="reactance (1/reactance) @ line [MW/rad]",
)


# Model decision variables
def supply_bounds(model: Model, o: int) -> tuple[float, float]:
    return (0.0, offers[o, "quantity"])


def flow_bounds(model: Model, ell: int) -> tuple[float, float]:
    capacity = lines[ell, "capacity"]
    return (-capacity, +capacity)


model.p = pyo.Var(Offers, bounds=supply_bounds, doc="power injection @ offer [MW]")
model.f = pyo.Var(Lines, bounds=flow_bounds, doc="power flow @ line [MW]")
model.theta = pyo.Var(
    Buses, bounds=(-math.pi, +math.pi), doc="voltage angle @ bus [rad]"
)
model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

# Model objective function
model.total_cost = pyo.Objective(
    expr=sum(offers[o, "price"] * model.p[o] for o in Offers),
    sense=pyo.minimize,
    doc="total operating cost [$/h]",
)

# Model constraints


def network_incidence(b: int, ell: int) -> int:
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


bus_to_line = np.array([[network_incidence(b, ell) for ell in Lines] for b in Buses])
bus_to_offer = np.array(
    [
        [any(supply_incidence(b, g, o) for g in Generators) for o in Offers]
        for b in Buses
    ]
)


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
            sign * model.reactances[b] * model.theta[b]
            for b in Buses
            if (sign := network_incidence(b, ell))
        )
        == model.f[ell]
    )


model.balance = pyo.Constraint(Buses, rule=balance_rule, doc="power balance @ bus")
model.flow = pyo.Constraint(Lines, rule=flow_rule, doc="power flow @ line")
model.reference_angle = pyo.Constraint(
    expr=model.theta[0] == 0.0, doc="reference voltage angle @ bus[0]"
)


# Optimization
def optimize(model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


total_cost = optimize(model, tee=True)  # tee output to console
load_payments = 0
generator_payments = 0

# Transfer solution to tables
quantity = [pyo.value(model.p[o]) for o in Offers]
flow = [pyo.value(model.f[ell]) for ell in Lines]
angles_deg = [pyo.value(model.theta[b]) * 180 / math.pi for b in Buses]
marginal_prices = [model.dual[model.balance[b]] for b in Buses]

# Extend data tables with decision variables
offers = offers.with_columns(quantity=pl.Series(quantity))
lines = lines.with_columns(flow=pl.Series(flow))
buses = buses.with_columns(
    angle_deg=pl.Series(angles_deg), price=pl.Series(marginal_prices)
)

# Extend data tables with post-processed quantities
lines = lines.with_columns(  # utilization of each line
    (pl.col("flow").abs() / pl.col("capacity")).alias("utilization")
)
offers = offers.join(  # bus_id for each offer
    generators[:, ("id", "bus_id")], left_on="generator_id", right_on="id", how="left"
)
offers = offers.with_columns(  # utilization of each offer
    (pl.col("dispatched") / pl.col("quantity")).alias("utilization")
)

print("offers -", offers)
print("lines -", lines)
print("buses -", buses)
print("generators -", generators)

# Evaluate marginal prices


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

# Helper functions for plotting


def mapvalues(f, keys, *values) -> dict:
    return dict(zip(keys, map(f, *values)))


def hybrid_layout(
    G: nx.Graph,
    hub_scale: float = 1.0,
    satellite_scale: float = 1.0,
    k: float = 0.4,
    seed: int = 0,
) -> dict:
    """Produces a better outcome than either pure layout provided by networkx"""
    hubs = [n for n in G.nodes if G.degree(n) > 1]
    pos = nx.kamada_kawai_layout(G.subgraph(hubs), scale=hub_scale)
    for hub in hubs:
        satellites = {
            *G.successors(hub),
            *G.predecessors(hub),
        }.difference(hubs)
        pos.update(
            nx.circular_layout(
                G.subgraph(satellites),
                center=pos[hub],
                scale=satellite_scale,
            )
        )
        print(f"{hub} => {satellites}")
    assert len(pos) == G.number_of_nodes()
    return nx.spring_layout(G, pos=pos, fixed=hubs, seed=seed, k=k)


# Network definition for plotting
network = nx.DiGraph()

# Node defnition
bus_price_labels = mapvalues(
    "{:}MW\n@ ${:.2f}/MWh".format, buses["id"], buses["load"], buses["price"]
)
offer_price_labels = mapvalues(
    "{:}MW\n@ ${:}/MWh".format, offers["id"], offers["quantity"], offers["price"]
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

# Edge defnition
flow_labels = mapvalues(
    "{:.0f}MW/\n{:.0f}MW".format,
    zip(lines["from_bus_id"], lines["to_bus_id"]),
    lines["flow"],
    lines["capacity"],
)
supply_labels = mapvalues(
    "{:.0f}MW/\n{:.0f}MW".format,
    zip(offers["id"], offers["bus_id"]),
    offers["dispatched"],
    offers["quantity"],
)
network.add_edges_from(zip(lines["from_bus_id"], lines["to_bus_id"]))
network.add_edges_from(zip(offers["id"], offers["bus_id"]))
edge_utilization = [*lines["utilization"], *offers["utilization"]]
edge_labels = flow_labels | supply_labels

# Network layout
scale = 50.0
pos: dict = hybrid_layout(
    network, hub_scale=scale, satellite_scale=scale, k=scale * 1.0
)

# Network annotation
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
    f"            Total Cost: ${total_cost:.2f}/h\n"
    f"    Payments from Load: ${load_payments:.2f}/h\n"
    f"Payments to Generators: ${generator_payments:.2f}/h\n"
)
plt.axis("off")
plt.savefig("network.png", dpi=300)
plt.show(block=False)

# Path analysis

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
                reactance, utilization = (
                    lines.filter(
                        (pl.col("from_bus_id") == from_node_id)
                        & (pl.col("to_bus_id") == to_node_id)
                    )
                    .select(["reactance", "utilization"])
                    .row(0)
                )
                path_resistance += 1 / reactance
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
                    reactance=1 / path_resistance,
                )
            )
            # for row in edge_rows:
            #     row["path_reactance"] = 1 / path_resistance
            # path_edge_rows.extend(edge_rows)

paths = pl.DataFrame(path_rows)
path_edges = pl.DataFrame(path_edge_rows)

group_keys = ["source", "target"]
grouped = paths.group_by(group_keys).agg(
    pl.col("reactance").sum().alias("total_reactance")
)
paths = paths.join(grouped, on=group_keys).with_columns(
    (pl.col("reactance") / pl.col("total_reactance")).alias("relative_reactance")
)

path_edges = path_edges.join(
    paths[:, ("path_id", "source", "target", "relative_reactance")], on=["path_id"]
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
    "target", "relative_reactance", "source"
).iter_rows():
    terms = [
        f"{c:.2f}*PTotal[{s}]" for c, s in zip(coeffs, sources) if s in input_nodes
    ]
    print(f"@ {target}: {' + '.join(terms)} == 0")

print("Constraints implied unit increment:")
for target, sources in path_summary.select("target", "source").iter_rows():
    terms = [f"PTotal[{s}]" for s in set(sources) if s in input_nodes]
    print(f"@ {target}: {' + '.join(terms)} == 1")
