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

FONT_SIZE: int = 8  # [points]
LABEL_OFFSET: float = -1.2  # times font_size


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


def label_nodes(
    fig: plt.Figure,
    ax: plt.Axes,
    pos: dict[str, np.ndarray],
    labels: dict[str, str],
    font_color: str,
    font_size: int,
    offset: float,
):
    offset = mpt.offset_copy(
        ax.transData, fig=fig, x=0, y=font_size * offset, units="points"
    )
    print(type(offset), offset)
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


def draw_nodes(
    ax: plt.Axes,
    network: nx.Graph,
    pos: dict[str, np.ndarray],
    nodelist: list[str],
    font_color: str,
    font_size: int,
    **kwargs,
):
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
    select: Callable[[float], bool],
    ax: plt.Axes,
    network: nx.Graph,
    pos: dict[str, np.ndarray],
    edge_labels: dict,
    weight: dict,
    font_color: str,
    font_size: int,
) -> dict:
    nx.draw_networkx_edge_labels(
        network,
        pos,
        edge_labels={
            key: value for key, value in edge_labels.items() if select(weight[*key])
        },
        font_color=font_color,
        font_size=font_size,
        bbox=dict(alpha=0.0),  # transparent background
        ax=ax,
    )


def make_network(
    buses: DataFrame,
    lines: DataFrame,
    offers: DataFrame,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
):
    """Network definition for plotting."""
    network = nx.DiGraph()
    network.add_nodes_from(buses["id"])
    network.add_nodes_from(offers["id"])
    network.add_edges_from(zip(lines["from_bus_id"], lines["to_bus_id"]))
    network.add_edges_from(zip(offers["id"], offers["bus_id"]))
    pos: dict = nx.nx_agraph.graphviz_layout(network.to_undirected())
    # Scale coordinates to properly fill the figure
    xy = np.vstack(tuple(pos.values()))
    pos = dict(zip(pos.keys(), normalize(xy, scale_x, scale_y)))
    return network, pos


def plot_network(
    fig: plt.Figure,
    ax: plt.Axes,
    network: nx.Graph,
    pos: dict[str, np.ndarray],
    buses: DataFrame,
    generators: DataFrame,
    offers: DataFrame,
    font_size: int = FONT_SIZE,  # [points]
    label_offset: float = LABEL_OFFSET,  # times font_size
):
    """Plot the problem data, dispatch instructions, and prices."""
    bus_load_norm = mc.Normalize(vmin=min(buses["load"]), vmax=max(buses["load"]))
    bus_load_cmap = cm.coolwarm  # coolwarm | inferno | magma | plasma etc.
    draw_nodes(
        nodelist=buses["id"],
        node_color=[bus_load_cmap(bus_load_norm(load)) for load in buses["load"]],
        ax=ax,
        network=network,
        pos=pos,
        font_color="black",
        font_size=font_size,
        node_shape="s",
        alpha=0.5,
    )
    generator_cmap = plt.get_cmap(name="tab10", lut=len(generators))
    generator_colors = {
        gid: mc.to_hex(generator_cmap(i)) for i, gid in enumerate(generators["id"])
    }
    draw_nodes(
        nodelist=offers["id"],
        node_color=[generator_colors[gid] for gid in offers["generator_id"]],
        font_color="black",
        font_size=font_size,
        node_shape="o",
        alpha=0.5,
        ax=ax,
        network=network,
        pos=pos,
    )
    label_nodes(
        labels=node_annotations(buses, "id", "{:.0f}MW", "load"),
        offset=label_offset,
        font_color="darkgreen",
        font_size=font_size,
        fig=fig,
        ax=ax,
        pos=pos,
    )
    label_nodes(
        labels=node_annotations(offers, "id", "≤{:}MW\n${:}/MWh", "quantity", "price"),
        offset=label_offset,
        font_color="black",
        font_size=font_size,
        fig=fig,
        ax=ax,
        pos=pos,
    )
    # No labels available at this point
    nx.draw_networkx_edges(network, pos, ax=ax)


def plot_opf(
    fig: plt.Figure,
    ax: plt.Axes,
    network: nx.Graph,
    pos: dict[str, np.ndarray],
    buses: DataFrame,
    lines: DataFrame,
    offers: DataFrame,
    font_size: int = FONT_SIZE,  # [points]
    label_offset: float = LABEL_OFFSET,  # times font_size
):
    epsilon = 1 / 100
    # fmt: off
    utilization = (
        dict(zip(zip(offers["id"], offers["bus_id"]), offers["utilization"])) |
        dict(zip(zip(lines["from_bus_id"], lines["to_bus_id"]), lines["utilization"]))
    )
    # fmt: on
    flow_edge_labels = edge_annotations(
        lines, "from_bus_id", "to_bus_id", "{:.0f}MW\n{:.0f}%", "flow", "utilization"
    )
    supply_edge_labels = edge_annotations(
        offers, "id", "bus_id", "{:.0f}MW", "dispatch"
    )
    edge_labels = flow_edge_labels | supply_edge_labels

    label_nodes(
        labels=node_annotations(buses, "id", "${:.2f}/MWh", "price"),
        offset=label_offset * 2,
        font_color="darkgreen",
        font_size=font_size,
        fig=fig,
        ax=ax,
        pos=pos,
    )
    nx.draw_networkx_edges(
        network,
        pos,
        edgelist=utilization.keys(),
        edge_color=utilization.values(),
        edge_vmin=0.0,
        edge_vmax=1.0,
        edge_cmap=cm.coolwarm,
        ax=ax,
    )
    draw_edge_labels(
        select=lambda u: u < epsilon,
        weight=utilization,
        edge_labels=edge_labels,
        ax=ax,
        network=network,
        pos=pos,
        font_color="black",
        font_size=font_size,
    )
    draw_edge_labels(
        select=lambda u: epsilon <= u <= 1.0 - epsilon,
        weight=utilization,
        edge_labels=edge_labels,
        ax=ax,
        network=network,
        pos=pos,
        font_color="blue",
        font_size=font_size,
    )
    draw_edge_labels(
        select=lambda u: 1.0 - epsilon < u,
        weight=utilization,
        edge_labels=edge_labels,
        ax=ax,
        network=network,
        pos=pos,
        font_color="red",
        font_size=font_size,
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
