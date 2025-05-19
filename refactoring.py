import dataclasses
import math
import textwrap
from typing import Callable, Optional, Self
import numpy as np
from numpy.typing import NDArray
import polars as pl
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.expr.relational_expr import EqualityExpression, InequalityExpression


@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class Serialization:

    def _generate(self, string: Callable, outer: str, inner: str) -> str:
        """Shared logic of __repr__ and __str__."""

        def indent(text: str) -> str:
            return textwrap.indent(text, " " * 4)

        body = "\n".join(
            inner.format(
                name=field.name, value=indent(string(getattr(self, field.name)))
            )
            for field in dataclasses.fields(self)
        )

        return outer.format(name=type(self).__name__, value=indent(body))

    def __repr__(self) -> str:
        """String representation."""
        return self._generate(
            repr,
            outer="{name}(\n{value}\n)",
            inner="{name}=\\\n{value},",
        )

    def __str__(self) -> str:
        """Pretty-print."""
        return self._generate(
            str,
            outer="{{\n{value}\n}}",
            inner="{name}:\n{value}",
        )


@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class Data(Serialization):
    bus_load: NDArray[np.float64]
    line_capacity: NDArray[np.float64]
    line_susceptance: NDArray[np.float64]
    offer_max_quantity: NDArray[np.float64]
    offer_price: NDArray[np.float64]
    line_bus_incidence: NDArray[np.int8]
    offer_bus_incidence: NDArray[np.int8]
    reference_bus: int

    def __post_init__(self) -> None:
        """Validate array dimensions."""
        assert len(self.line_capacity) == len(self.line_susceptance)
        assert len(self.offer_max_quantity) == len(self.offer_price)
        assert self.line_bus_incidence.shape == (
            len(self.line_capacity),
            len(self.bus_load),
        )
        assert self.offer_bus_incidence.shape == (
            len(self.offer_price),
            len(self.bus_load),
        )

    @classmethod
    def init(
        cls,
        buses: pl.DataFrame,
        demands: pl.DataFrame,
        generators: pl.DataFrame,
        lines: pl.DataFrame,
        offers: pl.DataFrame,
        reference_bus: int = 0,
    ) -> Self:
        """Initialize from dataframes."""

        bus_indices = range(buses.height)
        demand_indices = range(demands.height)
        generator_indices = range(generators.height)
        line_indices = range(lines.height)
        offer_indices = range(offers.height)

        def network_incidence(ell: int, b: int) -> int:
            if buses[b, "id"] == lines[ell, "from_bus_id"]:
                return -1
            if buses[b, "id"] == lines[ell, "to_bus_id"]:
                return +1
            else:
                return 0

        def supply_incidence(o: int, b: int, g: int) -> bool:
            bus_generator: bool = buses[b, "id"] == generators[g, "bus_id"]
            generator_offer: bool = generators[g, "id"] == offers[o, "generator_id"]
            return bus_generator and generator_offer

        def load_incidence(d: int, b: int) -> bool:
            if buses[b, "id"] == demands[d, "bus_id"]:
                return -1
            else:
                return 0

        line_bus_incidence = np.array(
            [[network_incidence(ell, b) for b in bus_indices] for ell in line_indices]
        )
        offer_bus_incidence = np.array(
            [
                [
                    any(supply_incidence(o, b, g) for g in generator_indices)
                    for b in bus_indices
                ]
                for o in offer_indices
            ],
            dtype=float,
        )

        load_bus_incidence = np.array(
            [[load_incidence(d, b) for b in bus_indices] for d in demand_indices],
            dtype=float,
        )
        demand_loads = demands[:, "load"].to_numpy()

        return cls(
            bus_load=demand_loads @ -load_bus_incidence,
            line_capacity=lines[:, "capacity"].to_numpy(),
            line_susceptance=lines[:, "susceptance"].to_numpy(),
            offer_max_quantity=offers[:, "max_quantity"].to_numpy(),
            offer_price=offers[:, "price"].to_numpy(),
            line_bus_incidence=line_bus_incidence,
            offer_bus_incidence=offer_bus_incidence,
            reference_bus=reference_bus,
        )


@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class Result(Serialization):
    total_cost: float
    dispatch_quantity: NDArray[np.float64]
    line_flow: NDArray[np.float64]
    voltage_angle: NDArray[np.float64]
    energy_price: Optional[NDArray[np.float64]] = None
    congestion_price: Optional[NDArray[np.float64]] = None
    loss_price: Optional[NDArray[np.float64]] = None


def objective_value(solver, model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.total_cost)


def formulate(data, solver, tee: bool = False) -> Model:
    """Formulate the optimization problem for the given power system data."""

    model = pyo.ConcreteModel()

    # Index sets
    bus_indices = range(data.bus_load.size)
    line_indices = range(data.line_bus_incidence.shape[0])
    offer_indices = range(data.offer_bus_incidence.shape[0])

    # Parameters for pricing
    model.loads = pyo.Param(
        bus_indices,
        initialize={b: data.bus_load[b] for b in bus_indices},
        mutable=True,
        doc="load (demand) @ bus [MW]",
    )
    model.line_capacities = pyo.Param(
        line_indices,
        initialize={ell: data.line_capacity[ell] for ell in line_indices},
        mutable=True,
        doc="capacity @ line [MW]",
    )

    # Decision variables & bounds
    def supply_bounds(model: Model, o: int) -> tuple[float, float]:
        return (0.0, data.offer_max_quantity[o])

    model.p = pyo.Var(
        offer_indices, bounds=supply_bounds, doc="power injection @ offer [MW]"
    )
    model.f = pyo.Var(line_indices, doc="power flow @ line [MW]")
    model.theta = pyo.Var(
        bus_indices, bounds=(-math.pi, +math.pi), doc="voltage angle @ bus [rad]"
    )
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Objective function
    model.total_cost = pyo.Objective(
        expr=sum(data.offer_price[o] * model.p[o] for o in offer_indices),
        sense=pyo.minimize,
        doc="total operating cost [$/h]",
    )

    def balance_rule(model: Model, b: int) -> EqualityExpression:
        return (
            sum(
                alpha * model.p[o]
                for o in offer_indices
                if (alpha := data.offer_bus_incidence[o, b]) != 0
            )
            + sum(
                beta * model.f[ell]
                for ell in line_indices
                if (beta := data.line_bus_incidence[ell, b]) != 0
            )
            == model.loads[b]
        )

    def flow_rule(model: Model, ell: int) -> EqualityExpression:
        return (
            sum(
                gamma * model.theta[b] * lines[ell, "susceptance"]
                for b in bus_indices
                if (gamma := data.line_bus_incidence[ell, b]) != 0
            )
            == model.f[ell]
        )

    # Equality constraints
    model.balance = pyo.Constraint(
        bus_indices, rule=balance_rule, doc="power balance @ bus"
    )
    model.flow = pyo.Constraint(line_indices, rule=flow_rule, doc="power flow @ line")
    model.reference_angle = pyo.Constraint(
        expr=model.theta[reference_bus] == 0, doc="reference voltage angle @ bus[0]"
    )

    def flow_bound_lower_rule(model: Model, ell: int) -> InequalityExpression:
        return -model.line_capacities[ell] <= model.f[ell]

    def flow_bound_upper_rule(model: Model, ell: int) -> InequalityExpression:
        return model.f[ell] <= model.line_capacities[ell]

    # Inequality constraints:
    # NB: Implemented as a general Constraint because
    # some solvers do not support sensitivity analysis on Var bounds
    model.flow_bounds_lower = pyo.Constraint(
        line_indices, rule=flow_bound_lower_rule, doc="flow lower bound @ line"
    )
    model.flow_bounds_upper = pyo.Constraint(
        line_indices, rule=flow_bound_upper_rule, doc="flow upper bound @ line"
    )

    # Solve optimization model
    total_cost = objective_value(solver, model, tee=tee)

    # Extract optimal decision values
    return Result(
        total_cost=total_cost,
        dispatch_quantity=np.array([pyo.value(model.p[o]) for o in offer_indices]),
        line_flow=np.array([pyo.value(model.f[ell]) for ell in line_indices]),
        voltage_angle=np.array([pyo.value(model.theta[b]) for b in bus_indices]),
        energy_price=np.array([model.dual[model.balance[b]] for b in bus_indices]),
        congestion_price=np.array(
            [
                model.dual[model.flow_bounds_upper[ell]]
                - model.dual[model.flow_bounds_lower[ell]]
                for ell in line_indices
            ]
        ),
    )


# Units of measure
Id = pl.String
MW = pl.Float64
MWPerRad = pl.Float64
USDPerMWh = pl.Float64

##############################################################################
# Data data
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


net = Data.init(
    buses=buses,
    demands=demands,
    generators=generators,
    lines=lines,
    offers=offers,
)
solver = pyo.SolverFactory("highs")
result = formulate(net, solver)
