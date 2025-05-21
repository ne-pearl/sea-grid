from typing import Any, Callable, Optional
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import networkx as nx
import numpy as np
import polars as pl
from datastructures import Data, Result


def augment(
    data: Data,
    result: Result,
    buses: pl.DataFrame,
    generators: pl.DataFrame,
    lines: pl.DataFrame,
    offers: pl.DataFrame,
    reference_bus: str,
) -> dict[str, Any]:
    buses = buses.with_columns(
        angle_deg=pl.Series(np.rad2deg(result.voltage_angle)),
        price=pl.Series(result.energy_price),
        load=pl.Series(data.bus_load),
    )
    lines = lines.with_columns(flow=pl.Series(result.line_flow)).with_columns(
        (pl.col("flow").abs() / pl.col("capacity") * 100).alias("utilization")
    )
    offers = (
        offers.with_columns(
            pl.arange(0, pl.len()).over("generator_id").alias("tranche")
        )
        .with_columns(
            pl.format("{}/{}", pl.col("generator_id"), pl.col("tranche") + 1).alias(
                "id"
            )
        )
        .join(
            generators[:, ("id", "bus_id")],
            left_on="generator_id",
            right_on="id",
            how="left",
        )
        .with_columns(pl.Series(result.dispatch_quantity).alias("quantity"))
        .with_columns(
            (pl.col("quantity") / pl.col("max_quantity") * 100).alias("utilization")
        )
    )
    return dict(
        buses=buses,
        generators=generators,
        lines=lines,
        offers=offers,
        reference_bus=reference_bus,
    )


def hybrid_layout(
    graph: nx.Graph,
    scale: float = 1.0,
    kscale: float = 1.0,
    seed: int = 0,
    **kwargs,
) -> dict:
    """
    This hybrid layout seems to produce better outcome than the pure layouts provided by networkx.
    """
    # Work-around: spring_layout allows orientation to influence layout in evil ways
    assert not graph.is_directed()
    hubs = [n for n in graph.nodes if graph.degree(n) > 1]
    pos = nx.kamada_kawai_layout(graph.subgraph(hubs), scale=scale)
    for hub in hubs:
        satellites = {*graph.neighbors(hub)}.difference(hubs)
        satpos = nx.circular_layout(
            graph.subgraph(satellites),
            center=pos[hub],
            scale=scale,
        )
        pos.update(satpos)
    assert len(pos) == graph.number_of_nodes()
    k = kscale / np.sqrt(len(graph))
    return nx.spring_layout(graph, pos=pos, fixed=hubs, k=k, seed=seed, **kwargs)


def node_annotations(
    df: pl.DataFrame, key: str, template: str, *value: str
) -> dict[str, str]:
    """Create label dictionary from value sequences."""
    keys = df[:, key]
    values = map(template.format, *df[:, value])
    return dict(zip(keys, values))


def edge_annotations(
    df: pl.DataFrame, tail_key: str, head_key: str, template: str, *value: str
) -> dict[str, str]:
    """Create label dictionary from value sequences."""
    keys = zip(df[:, tail_key], df[:, head_key])
    values = map(template.format, *df[:, value])
    return dict(zip(keys, values))


def plot_tables(
    buses: pl.DataFrame,
    lines: pl.DataFrame,
    generators: pl.DataFrame,
    offers: pl.DataFrame,
    kscale: float,
    dy: int = -2,  # [points]
    font_size: int = 8,  # [points]
    iterations: int = 50000,
    **_,
) -> None:
    """Plot the problem data, dispatch instructions, and prices."""

    # Network definition for plotting
    network = nx.DiGraph()

    network.add_nodes_from(buses["id"])
    network.add_nodes_from(offers["id"])

    line_edges = list(zip(lines[:, "from_bus_id"], lines[:, "to_bus_id"]))
    line_utilization = lines[:, "utilization"].to_list()
    network.add_weighted_edges_from(
        zip(*zip(*line_edges), line_utilization),
        weight="utilization",
        tag="lines",
    )

    offer_edges = list(zip(offers[:, "id"], offers[:, "bus_id"]))
    offer_utilization = offers[:, "utilization"].to_list()
    network.add_weighted_edges_from(
        zip(*zip(*offer_edges), offer_utilization),
        weight="utilization",
        tag="offers",
    )

    # Network layout
    pos: dict = hybrid_layout(
        network.to_undirected(), iterations=iterations, kscale=kscale
    )

    fig, ax = plt.subplots(figsize=np.array([15, 9]) * 0.9, dpi=150, frameon=False)
    ax.axis("equal")

    def draw_nodes(nodelist, font_color="black", **kwargs):
        nonlocal ax, network, pos, font_size
        nx.draw_networkx_nodes(network, pos, nodelist=nodelist, ax=ax, **kwargs)
        nx.draw_networkx_labels(
            network,
            pos,
            labels={id: id for id in nodelist},
            font_size=font_size,
            font_color=font_color,
            ax=ax,
        )

    def draw_edge_labels(
        selected: Callable[[float], bool],
        edge_labels: dict,
        utilization: dict,
        font_color: str,
    ) -> dict:
        nonlocal ax, network, pos, font_size
        nx.draw_networkx_edge_labels(
            network,
            pos,
            edge_labels={
                key: value
                for key, value in edge_labels.items()
                if selected(utilization[*key])
            },
            font_color=font_color,
            font_size=font_size,
            ax=ax,
        )

    bus_load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
    bus_load_cmap = cm.coolwarm  # coolwarm | inferno | magma | plasma etc.
    draw_nodes(
        buses["id"],
        node_color=[bus_load_cmap(bus_load_norm(load)) for load in buses["load"]],
        node_shape="s",
    )

    generator_cmap = plt.get_cmap(name="tab10", lut=generators.height)
    generator_colors = {
        gid: mc.to_hex(generator_cmap(i)) for i, gid in enumerate(generators["id"])
    }
    draw_nodes(
        offers["id"],
        node_color=[generator_colors[gid] for gid in offers["generator_id"]],
        node_shape="o",
    )

    def label_nodes(labels: dict, font_color: str):
        nonlocal pos, font_size
        offset = mpt.offset_copy(
            ax.transData,
            fig=fig,
            x=0,
            y=-(font_size + 2),
            units="points",
        )
        for node, (x, y) in pos.items():
            label = labels.get(node)
            if label is not None:
                ax.text(
                    x,
                    y,
                    label,
                    transform=offset,
                    fontsize=font_size,
                    color=font_color,
                    ha="center",
                    va="top",
                )

    bus_price_node_labels = node_annotations(buses, "id", "${:.2f}/MWh", "price")
    label_nodes(bus_price_node_labels, font_color="darkgreen")

    offer_price_node_labels = node_annotations(
        offers, "id", "≤{:}MW\n@ ${:}/MWh", "max_quantity", "price"
    )
    label_nodes(offer_price_node_labels, font_color="black")

    # NetworkX does not preserve edge insertion order, so reconstruct a consistent ordering
    utilization = nx.get_edge_attributes(network, "utilization")

    nx.draw_networkx_edges(
        network,
        pos,
        edgelist=utilization.keys(),
        edge_color=utilization.values(),
        edge_cmap=cm.coolwarm,
        edge_vmin=0.0,
        edge_vmax=100.0,
        ax=ax,
    )

    flow_edge_labels = edge_annotations(
        lines, "from_bus_id", "to_bus_id", "{:.0f}MW\n{:.0f}%", "flow", "utilization"
    )
    supply_edge_labels = edge_annotations(
        offers, "id", "bus_id", "{:.0f}MW", "quantity"
    )
    edge_labels = flow_edge_labels | supply_edge_labels

    epsilon = 1.0  # [%]
    draw_edge_labels(
        lambda percent: 0.0 <= percent < epsilon,
        edge_labels=edge_labels,
        utilization=utilization,
        font_color="black",
    )
    draw_edge_labels(
        lambda percent: epsilon <= percent <= 100.0 - epsilon,
        edge_labels=edge_labels,
        utilization=utilization,
        font_color="blue",
    )
    draw_edge_labels(
        lambda percent: 100.0 - epsilon < percent,
        edge_labels=edge_labels,
        utilization=utilization,
        font_color="red",
    )

    total_cost = sum(offers[:, "quantity"] * offers[:, "price"])
    load_payment = sum(buses[:, "load"] * buses[:, "price"])

    # Display network
    fig.text(
        0.95,
        0.05,  # figure coordinates (0–1)
        f"    From Load: ${load_payment:.2f}/h\n"
        f"To Generators: ${total_cost:.2f}/h\n"
        f"   Difference: ${load_payment - total_cost:.2f}/h",
        va="bottom",
        ha="right",
        fontsize=font_size,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, ax


def plot(
    data: Data,
    result: Result,
    kscale: float = 1.0,
    **tables,
):
    augmented: dict[str, Any] = augment(data=data, result=result, **tables)
    return plot_tables(**augmented, kscale=kscale)
