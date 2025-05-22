import pprint
from typing import Any, Callable
import matplotlib.cm as cm
import matplotlib.colors as mc
import matplotlib.pyplot as plt
import matplotlib.transforms as mpt
import networkx as nx
import numpy as np
from pandas import DataFrame
from datastructures import Data, Result


def normalize(xy: np.ndarray, scale_x: float = 1.0, scale_y: float = 1.0) -> np.ndarray:
    """Shift and scale points to [-1,1]^2, with optional rescaling."""
    assert xy.ndim == 2 and xy.shape[1] == 2
    centered = xy - np.median(xy, axis=0)
    normalized = centered / np.max(np.abs(centered))
    return normalized * np.array([scale_x, scale_y])


def node_annotations(
    df: DataFrame, key: str, template: str, *value: str
) -> dict[str, str]:
    """Create label dictionary from value sequences."""
    keys = df[key]
    values = list(map(template.format, *df[list(value)].to_numpy().T))
    return dict(zip(keys, values))


def edge_annotations(
    df: DataFrame, tail_key: str, head_key: str, template: str, *value: str
) -> dict[str, str]:
    """Create label dictionary from value sequences."""
    keys = zip(df[tail_key], df[head_key])
    values = list(map(template.format, *df[list(value)].to_numpy().T))
    return dict(zip(keys, values))


def plot_tables(
    buses: DataFrame,
    generators: DataFrame,
    lines: DataFrame,
    offers: DataFrame,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
    font_size: int = 8,  # [points]
    **_,
) -> None:
    """Plot the problem data, dispatch instructions, and prices."""

    # Network definition for plotting
    network = nx.DiGraph()

    # Nodes
    network.add_nodes_from(buses["id"])
    network.add_nodes_from(offers["id"])

    # Edges
    line_edges = list(zip(lines["from_bus_id"], lines["to_bus_id"]))
    line_utilization = lines["utilization"].to_list()
    network.add_weighted_edges_from(
        zip(*zip(*line_edges), line_utilization),
        weight="utilization",
        tag="lines",
    )
    offer_edges = list(zip(offers["id"], offers["bus_id"]))
    offer_utilization = offers["utilization"].to_list()
    network.add_weighted_edges_from(
        zip(*zip(*offer_edges), offer_utilization),
        weight="utilization",
        tag="offers",
    )

    # Node positions
    # Undirected since some layout algorithms perform poorly with directed graphs
    pos: dict = nx.nx_agraph.graphviz_layout(network.to_undirected())

    # Scale coordinates to properly fill the figure
    xy = np.vstack(tuple(pos.values()))
    pos = dict(zip(pos.keys(), normalize(xy, scale_x, scale_y)))

    # Create blank figure and plotting axes
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
            bbox=dict(alpha=0.0),  # transparent background
            ax=ax,
        )

    bus_load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
    bus_load_cmap = cm.coolwarm  # coolwarm | inferno | magma | plasma etc.
    draw_nodes(
        buses["id"],
        node_color=[bus_load_cmap(bus_load_norm(load)) for load in buses["load"]],
        node_shape="s",
        alpha=0.5,
    )

    generator_cmap = plt.get_cmap(name="tab10", lut=len(generators))
    generator_colors = {
        gid: mc.to_hex(generator_cmap(i)) for i, gid in enumerate(generators["id"])
    }
    draw_nodes(
        offers["id"],
        node_color=[generator_colors[gid] for gid in offers["generator_id"]],
        node_shape="o",
        alpha=0.5,
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

    bus_price_node_labels = node_annotations(
        buses, "id", "{:.0f}MW\n${:.2f}/MWh", "load", "price"
    )
    label_nodes(bus_price_node_labels, font_color="darkgreen")

    offer_price_node_labels = node_annotations(
        offers, "id", "≤{:}MW\n${:}/MWh", "quantity", "price"
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
        offers, "id", "bus_id", "{:.0f}MW", "dispatch"
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

    total_cost = sum(offers["dispatch"] * offers["price"])
    load_payment = sum(buses["load"] * buses["price"])

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


def display(
    data: Data,
    result: Result,
    font_size: int,
    scale_x: float,
    scale_y: float,
    **tables,
):
    augmented: dict = augment(data=data, result=result, **tables)
    pprint.pprint(augmented)
    fig, ax = plot_tables(
        **augmented,
        font_size=font_size,
        scale_x=scale_x,
        scale_y=scale_y,
    )
    return augmented, fig, ax
