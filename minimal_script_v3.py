import cvxpy as cp
import numpy as np
import sea_incidence as incidence
from triangle2_pandas import buses, generators, lines, offers, reference_bus, base_power

line_bus = incidence.line_bus(buses=buses, lines=lines)
offer_bus = incidence.offer_bus(offers=offers, buses=buses, generators=generators)
reference_bus_index = incidence.reference_bus(buses, reference_bus)
offers = offers.merge(generators, left_on="generator_id", right_on="id")

p = cp.Variable(len(offers), name="p")  # generation
f = cp.Variable(len(lines), name="f")  # line flows
θ = cp.Variable(len(buses), name="θ")  # bus angles

balance_constraints = [
    cp.sum([p[o] * offer_bus[o, b] for o in offers.index])
    + cp.sum([f[ell] * line_bus[ell, b] for ell in lines.index])
    == buses.at[b, "load"]
    for b in buses.index
]
flow_constraints = [
    f[ell]
    == cp.sum([line_bus[ell, b] * θ[b] for b in buses.index])
    * base_power
    / lines.at[ell, "reactance"]
    for ell in lines.index
]
objective = cp.Minimize(cp.sum([offers.at[o, "price"] * p[o] for o in offers.index]))
problem = cp.Problem(
    objective,
    [
        *balance_constraints,
        *flow_constraints,
        θ[reference_bus_index] == 0,
        f >= -lines["capacity"],
        f <= lines["capacity"],
        p >= 0,
        p <= offers["quantity"],
    ],
)
problem.solve(solver="ECOS")

offers["dispatch"] = p.value
lines["flow"] = f.value
buses["angle"] = θ.value
buses["lmp_total"] = [-c.dual_value for c in balance_constraints]

print(problem)
print(f"Optimal dispatch cost: ${problem.value:.2f} / h")

# =========================================================================

offer_dispatch = offers["dispatch"]
line_flow = lines["flow"]
bus_angle = buses["angle"]
line_reactance = lines["reactance"]

assert np.allclose(
    (line_bus @ bus_angle) * base_power / line_reactance,
    line_flow,
)
free_bus_ids = [b for b in buses.index if b != reference_bus_index]
K = line_bus[:, free_bus_ids]
KtB = K.T @ np.diag(1.0 / line_reactance)
SF = np.linalg.solve(KtB @ K, -KtB).T
bus_injections = offer_dispatch @ offer_bus - buses["load"]
assert np.allclose(bus_injections, -line_flow @ line_bus)
assert np.allclose(SF @ bus_injections[free_bus_ids], line_flow)
assert np.allclose(
    (SF @ offer_bus[:, free_bus_ids].T) @ offer_dispatch
    - SF @ buses.loc[free_bus_ids, "load"],
    line_flow,
)

p = cp.Variable(len(offers), name="p")  # generation
f = cp.Variable(len(lines), name="f")  # line flows

balance_constraint = cp.sum(p) == sum(buses["load"])

A = SF @ offer_bus[:, free_bus_ids].T
L = SF @ buses.loc[free_bus_ids, "load"]
flow_constraints = [
    cp.sum([A[ell, o] * p[o] for o in offers.index]) - L[ell] == f[ell]
    for ell in lines.index
]
flow_lower_bounds = f >= -lines["capacity"]
flow_upper_bounds = f <= lines["capacity"]
objective = cp.Minimize(cp.sum([offers.at[o, "price"] * p[o] for o in offers.index]))
problem = cp.Problem(
    objective,
    [
        balance_constraint,
        *flow_constraints,
        flow_lower_bounds,
        flow_upper_bounds,
        p >= 0,
        p <= offers["quantity"],
    ],
)
problem.solve(solver="ECOS")
energy_price = -balance_constraint.dual_value
mu_lower = -flow_lower_bounds.dual_value
mu_upper = -flow_upper_bounds.dual_value
buses["lmp_energy"] = energy_price
buses["lmp_congestion"] = 0.0
buses.loc[free_bus_ids, "lmp_congestion"] = SF.T @ (mu_upper - mu_lower)

assert np.allclose(p.value, offers["dispatch"])
assert np.allclose(f.value, lines["flow"])
print(f"Optimal dispatch cost: ${problem.value:.2f} / h")
