# %% [markdown]
# External dependencies

# %%
import cvxpy as cp

# %% [markdown]
# Problem data

# %%
from pjm5bus_pandas import buses, generators, lines, offers, reference_bus

base_power = 100  # [MVA]

# %% [markdown]
# Build index mappings

# %%
bus_idx = {bus_id: i for i, bus_id in enumerate(buses["id"])}
offers = offers.merge(generators, left_on="generator_id", right_on="id")

# %% [markdown]
# Decision variables

# %%
p = cp.Variable(len(offers), name="p")  # generation
f = cp.Variable(len(lines), name="f")  # line flows
θ = cp.Variable(len(buses), name="θ")  # bus angles


# %% [markdown]
# Power balance constraint at each bus

# %%
balance_constraints = []
for i, row in buses.iterrows():
    bus_id = row["id"]
    offers_in = offers[offers["bus_id"] == bus_id].index
    lines_in = lines[lines["to_bus_id"] == bus_id].index
    lines_out = lines[lines["from_bus_id"] == bus_id].index
    balance_constraints.append(
        cp.sum(p[offers_in]) + cp.sum(f[lines_in])
        == buses.at[i, "load"] + cp.sum(f[lines_out])
    )

# %% [markdown]
# Power flow constraint on each line

# %%
flow_constraints = []
for i, row in lines.iterrows():
    bus_from = bus_idx[row["from_bus_id"]]
    bus_to = bus_idx[row["to_bus_id"]]
    reactance = row["reactance"]
    flow_constraints.append(f[i] == (θ[bus_from] - θ[bus_to]) * base_power / reactance)

# %% [markdown]
# Objective function

# %%
objective = cp.Minimize(
    cp.sum([offer["price"] * p[o] for o, offer in offers.iterrows()])
)

# %% [markdown]
# Solve

# %%
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

# %% [markdown]
# Attach results

# %%
offers["dispatch"] = p.value
lines["flow"] = f.value
buses["angle"] = θ.value
buses["price"] = [c.dual_value for c in balance_constraints]

print(f"Optimal dispatch cost: ${problem.value:.2f} / h")
