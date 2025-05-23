from pandas import DataFrame
import cvxpy as cp


def dcopf(
    buses: DataFrame,
    generators: DataFrame,
    lines: DataFrame,
    offers: DataFrame,
    reference_bus: str,
    base_power: float = 100,
) -> float:
    """
    Solves a DC-OPF problem for the given data.
    Dataframe arguments are updated with optimal decision variables
    Returns the optimal objective value (total generation cost in $/h).
    """

    # Build index mappings
    bus_idx = {bus_id: i for i, bus_id in enumerate(buses["id"])}
    offers_ = offers.merge(generators, left_on="generator_id", right_on="id")

    # Decision variables
    p = cp.Variable(len(offers_), name="p")  # generation
    f = cp.Variable(len(lines), name="f")  # line flows
    θ = cp.Variable(len(buses), name="θ")  # bus angles

    # Power balance at each bus
    bus_balance: list[cp.Constraint] = []
    for i, row in buses.iterrows():
        bus_id = row["id"]
        offers_in = offers_[offers_["bus_id"] == bus_id].index
        lines_in = lines[lines["to_bus_id"] == bus_id].index
        lines_out = lines[lines["from_bus_id"] == bus_id].index
        bus_balance.append(
            cp.sum(p[offers_in]) + cp.sum(f[lines_in])
            == buses.at[i, "load"] + cp.sum(f[lines_out])
        )

    line_flow: list[cp.Constraint] = []
    for i, row in lines.iterrows():
        bus_from = bus_idx[row["from_bus_id"]]
        bus_to = bus_idx[row["to_bus_id"]]
        reactance = row["reactance"]
        line_flow.append(f[i] == (θ[bus_from] - θ[bus_to]) * base_power / reactance)

    objective = cp.Minimize(
        cp.sum([offer["price"] * p[o] for o, offer in offers_.iterrows()])
    )

    problem = cp.Problem(
        objective,
        [
            *bus_balance,
            *line_flow,
            f >= -lines["capacity"],
            f <= lines["capacity"],
            p >= 0,
            p <= offers_["quantity"],
            θ[bus_idx[reference_bus]] == 0,
        ],
    )
    problem.solve(solver="ECOS")

    # Attach results
    buses["angle"] = θ.value
    buses["price"] = [-c.dual_value for c in bus_balance]
    lines["flow"] = f.value
    offers["dispatch"] = p.value

    return problem.value


def postprocess(
    buses: DataFrame, generators: DataFrame, lines: DataFrame, offers: DataFrame
) -> None:
    """Extend data with additional benchmark fields."""
    agg = offers.groupby("generator_id").agg({"dispatch": "sum", "quantity": "sum"})
    generators["dispatch"] = generators["id"].map(agg["dispatch"])
    generators["capacity"] = generators["id"].map(agg["quantity"])
    generators["utilization"] = generators["dispatch"] / generators["capacity"]
    generators["revenue"] = generators["dispatch"] * generators["bus_id"].map(
        buses.set_index("id")["price"]
    )
    lines["utilization"] = lines["flow"].abs() / lines["capacity"]
    offers["utilization"] = offers["dispatch"] / offers["quantity"]
    offers["bus_id"] = offers["generator_id"].map(generators.set_index("id")["bus_id"])
    offers["tranche"] = offers.groupby("generator_id").cumcount() + 1
    offers["id"] = offers["generator_id"] + "/" + offers["tranche"].astype(str)
