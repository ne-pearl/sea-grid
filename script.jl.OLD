using DataFrames: DataFrame
using JuMP, HiGHS
using Graphs, Plots, GraphRecipes

axis(a) = axes(a, 1)

base_power = 1000

# Network data
nodes = [
    (id=1, load=0, x=-1, y=1),
    (id=2, load=0, x=1, y=1),
    (id=3, load=150, x=0, y=0),
] |> DataFrame

generators = [
    (id=1, node_id=1, capacity=200, cost=10),
    (id=2, node_id=2, capacity=200, cost=12),
    (id=3, node_id=1, capacity=200, cost=14),
] |> DataFrame

lines = [
    (id=1, from_node_id=1, to_node_id=2, susceptance=0.25, capacity=30),
    (id=2, from_node_id=1, to_node_id=3, susceptance=0.25, capacity=300),
    (id=3, from_node_id=2, to_node_id=3, susceptance=0.25, capacity=300),
] |> DataFrame

offers = [
    (id=1, generator_id=1, max_quantity=200, price=10),
    (id=2, generator_id=2, max_quantity=200, price=12),
    (id=3, generator_id=3, max_quantity=200, price=14),
] |> DataFrame

node_line = [
    if node.id == line.from_node_id
        -1
    elseif node.id == line.to_node_id
        +1
    else
        0
    end
    for node = eachrow(nodes), line = eachrow(lines)
]

node_generator_offer = [
    if node.id == generator.node_id && generator.id == offer.generator_id
        1
    else
        0
    end
    for node = eachrow(nodes), generator = eachrow(generators), offer = eachrow(offers)
]

model = Model(HiGHS.Optimizer)

@variable(model, p[o=axis(offers)])
@variable(model, f[ℓ=axis(lines)])
@variable(model, θ[n=axis(nodes)])

@objective(model, Min, sum(offers[:, :price] .* p))

@constraint(
    model,
    balance[n=axis(nodes)],
    sum(
        node_generator_offer[n, g, o] * p[o]
        for g = axis(generators)
        for o = axis(offers)
    ) +
    sum(node_line[n, ℓ] * f[ℓ] for ℓ = axis(lines))
    ==
    nodes[n, :load]
)

@constraint(
    model,
    flow[ℓ=axis(lines)],
    sum(base_power * node_line[n, ℓ] * θ[n] * lines[ℓ, :susceptance] for n = axis(nodes)) == f[ℓ]
)

@constraint(model, offer_bounds[o=axis(offers)], 0 .≤ p[o] .≤ offers[o, :max_quantity])
@constraint(model, flow_bounds[ℓ=axis(lines)], -lines[ℓ, :capacity] ≤ f[ℓ] ≤ +lines[ℓ, :capacity])
@constraint(model, angle_bounds[n=axis(nodes)], -π ≤ θ[n] ≤ +π)
@constraint(model, reference_angle, θ[1] == 0)

model

optimize!(model)

@show p_star = value.(p)
@show f_star = value.(f)
@show θ_star_deg = value.(θ) .* 180 ./ π
@show prices = dual.(balance)

# Select backend: https://gr-framework.org/
Plots.gr()


num_nodes = size(nodes, 1)
num_generators = size(generators, 1)
g = Graphs.DiGraph(num_nodes + num_generators)
edgelabel = Dict()
vertexlabel = Dict()

for node = eachrow(nodes)
    vertexlabel[length(vertexlabel)] = "node $(node.id)"
end

for generator = eachrow(generators)
    vertexlabel[length(vertexlabel)] = "gen $(generator.id)"
end

for line = eachrow(lines)
    Graphs.add_edge!(g, line.from_node_id, line.to_node_id)
    edgelabel[line.from_node_id, line.to_node_id] = "line $(line.id)"
end

for generator = eachrow(generators)
    vertex_id = num_nodes + generator.id
    Graphs.add_edge!(g, vertex_id, generator.node_id)
    edgelabel[vertex_id, generator.node_id] = ""
end

plot = GraphRecipes.graphplot(
    g;
    names=1:nv(g),
    edgelabel=edgelabel,
    vertexlabel=vertexlabel,
    method=:spring,
    nodeshape=:circle,
    nodesize=0.3,
    nodecolor=:cornflowerblue,
    fontsize=12
)

display(plot)
