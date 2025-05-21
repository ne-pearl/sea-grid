import numpy as np
import pyomo.environ as pyo
from pyomo.core import Model
from pyomo.core.base.param import ParamData
from pyomo.core.expr.relational_expr import EqualityExpression, InequalityExpression
from datastructures import Data, Result


def optimize(model, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.objective)


def formulate(data: Data, tee: bool = False) -> Result:
    """Formulate the optimization problem for the given power system data."""

    model = pyo.ConcreteModel()

    # Index sets
    bus_indices = range(data.bus_load.size)
    line_indices = range(data.line_bus_incidence.shape[0])
    offer_indices = range(data.offer_bus_incidence.shape[0])

    # Parameters for pricing
    model.bus_loads = pyo.Param(
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

    model.p = pyo.Var(offer_indices, bounds=supply_bounds, doc="dispatch @ offer [MW]")
    model.f = pyo.Var(line_indices, doc="power flow @ line [MW]")
    model.theta = pyo.Var(bus_indices, doc="voltage angle @ bus [rad]")

    # For objective sentivity to constraint parameters
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Objective function
    model.objective = pyo.Objective(
        expr=sum(data.offer_price[o] * model.p[o] for o in offer_indices),
        sense=pyo.minimize,
        doc="total operating cost [$/h]",
    )

    def balance_rule(model: Model, b: int) -> EqualityExpression:
        return (
            sum(
                binary * model.p[o]
                for o in offer_indices
                if (binary := data.offer_bus_incidence[o, b]) != 0
            )
            + sum(
                sign * model.f[ell]
                for ell in line_indices
                if (sign := data.line_bus_incidence[ell, b]) != 0
            )
            == model.bus_loads[b]
        )

    def flow_rule(model: Model, ell: int) -> EqualityExpression:
        return (
            model.f[ell]
            == sum(
                sign * model.theta[b]
                for b in bus_indices
                if (sign := data.line_bus_incidence[ell, b]) != 0
            )
            * data.base_power
            / data.line_reactance[ell]
        )

    # Equality constraints
    model.balance = pyo.Constraint(
        bus_indices, rule=balance_rule, doc="power balance @ bus"
    )
    model.flow = pyo.Constraint(line_indices, rule=flow_rule, doc="power flow @ line")
    model.reference_angle = pyo.Constraint(
        expr=model.theta[data.reference_bus] == 0,
        doc="reference voltage angle @ bus[0]",
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
    total_cost = optimize(model, tee=tee)

    # Extract optimal decision values
    return Result(
        model=model,
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


def marginal_price(model, parameter: ParamData, delta=1.0) -> float:
    original = parameter.value
    unperturbed = optimize(model)
    parameter.value += delta
    perturbed = optimize(model)
    parameter.value = original  # restore
    return (perturbed - unperturbed) / delta
