import cvxpy as cp
import sea_incidence as incidence
from pjm5bus_pandas import buses, generators, lines, offers, reference_bus, base_power

line_bus = incidence.line_bus(buses=buses, lines=lines)
offer_bus = incidence.offer_bus(offers=offers, buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)
bus_idx = {bus_id: i for i, bus_id in enumerate(buses["id"])}
offers = offers.merge(generators, left_on="generator_id", right_on="id")

p = cp.Variable(len(offers), name="p")  # generation
f = cp.Variable(len(lines), name="f")  # line flows
θ = cp.Variable(len(buses), name="θ")  # bus angles

balance_constraints = [
    cp.sum(p @ offer_bus[:, b]) + cp.sum(f @ line_bus[:, b]) == buses.at[b, "load"]
    for b in buses.index
]
flow_constraints = [
    f[ell] == (line_bus[ell, :] @ θ) * base_power / lines.at[ell, "reactance"]
    for ell in lines.index
]
objective = cp.Minimize(
    cp.sum([offer["price"] * p[o] for o, offer in offers.iterrows()])
)
problem = cp.Problem(
    objective,
    [
        *balance_constraints,
        *flow_constraints,
        f >= -lines["capacity"],
        f <= lines["capacity"],
        p >= 0,
        p <= offers["quantity"],
        θ[bus_idx[reference_bus]] == 0,
    ],
)
problem.solve(solver="ECOS")

offers["dispatch"] = p.value
lines["flow"] = f.value
buses["angle"] = θ.value
buses["price"] = [c.dual_value for c in balance_constraints]

print(problem)
print(f"Optimal dispatch cost: ${problem.value:.2f} / h")
