import numpy as np
import pyomo.environ as pyo
from datastructures import Data, Result


def optimize(model: pyo.ConcreteModel, **kwargs) -> float:
    """Solve the optimization problem encoded in model and return the objective value."""
    solver = pyo.SolverFactory("highs")
    results = solver.solve(model, **kwargs)
    assert results.solver.termination_condition is pyo.TerminationCondition.optimal
    return pyo.value(model.obj)


def marginal_price(model: pyo.ConcreteModel, parameter, delta=1.0) -> float:
    original = parameter.value
    unperturbed = optimize(model)
    parameter.value += delta
    perturbed = optimize(model)
    parameter.value = original  # restore
    return (perturbed - unperturbed) / delta


def get_price(model: pyo.ConcreteModel, constraint) -> float:
    return model.dual.get(constraint, np.nan)


def formulate(data: Data, tee: bool = False) -> Result:
    """Formulate the optimization problem for the given power system data."""

    model = pyo.ConcreteModel()

    model.Buses = pyo.Set(initialize=range(data.bus_load.size))
    model.Lines = pyo.Set(initialize=range(data.line_bus_incidence.shape[0]))
    model.Offers = pyo.Set(initialize=range(data.offer_bus_incidence.shape[0]))

    model.bus_loads = pyo.Param(
        model.Buses,
        initialize={b: data.bus_load[b] for b in model.Buses},
        mutable=True,
        doc="load (demand) @ bus [MW]",
    )
    model.line_capacities = pyo.Param(
        model.Lines,
        initialize={ell: data.line_capacity[ell] for ell in model.Lines},
        mutable=True,
        doc="capacity @ line [MW]",
    )

    # Decision variables & bounds
    def supply_bounds(model: pyo.ConcreteModel, o: int):
        return (0.0, data.offer_quantity[o])

    model.p = pyo.Var(model.Offers, bounds=supply_bounds, doc="dispatch @ offer [MW]")
    model.f = pyo.Var(model.Lines, doc="power flow @ line [MW]")
    model.theta = pyo.Var(model.Buses, doc="voltage angle @ bus [rad]")

    # For objective sentivity to constraint parameters
    model.dual = pyo.Suffix(direction=pyo.Suffix.IMPORT)

    # Objective function
    model.obj = pyo.Objective(
        expr=sum(data.offer_price[o] * model.p[o] for o in model.Offers),
        sense=pyo.minimize,
        doc="total operating cost [$/h]",
    )

    def balance_rule(model: pyo.ConcreteModel, b: int):
        return (
            sum(
                binary * model.p[o]
                for o in model.Offers
                if (binary := data.offer_bus_incidence[o, b]) != 0
            )
            + sum(
                sign * model.f[ell]
                for ell in model.Lines
                if (sign := data.line_bus_incidence[ell, b]) != 0
            )
            == model.bus_loads[b]
        )

    def flow_rule(model: pyo.ConcreteModel, ell: int):
        return (
            model.f[ell]
            == sum(
                sign * model.theta[b]
                for b in model.Buses
                if (sign := data.line_bus_incidence[ell, b]) != 0
            )
            * data.base_power
            / data.line_reactance[ell]
        )

    # Equality constraints
    model.balance = pyo.Constraint(
        model.Buses, rule=balance_rule, doc="power balance @ bus"
    )
    model.flow = pyo.Constraint(model.Lines, rule=flow_rule, doc="power flow @ line")
    model.reference_angle = pyo.Constraint(
        expr=model.theta[data.reference_bus] == 0,
        doc="reference voltage angle @ bus[0]",
    )

    def flow_bound_lower_rule(model: pyo.ConcreteModel, ell: int):
        return -model.line_capacities[ell] <= model.f[ell]

    def flow_bound_upper_rule(model: pyo.ConcreteModel, ell: int):
        return model.f[ell] <= model.line_capacities[ell]

    # Inequality constraints:
    # NB: Implemented as a general Constraint because
    # some solvers do not support sensitivity analysis on Var bounds
    model.flow_bounds_lower = pyo.Constraint(
        model.Lines, rule=flow_bound_lower_rule, doc="flow lower bound @ line"
    )
    model.flow_bounds_upper = pyo.Constraint(
        model.Lines, rule=flow_bound_upper_rule, doc="flow upper bound @ line"
    )

    # Solve optimization model
    total_cost = optimize(model, tee=tee)

    # Extract optimal decision values
    return Result(
        model=model,
        total_cost=total_cost,
        dispatch_quantity=np.array([pyo.value(model.p[o]) for o in model.Offers]),
        line_flow=np.array([pyo.value(model.f[ell]) for ell in model.Lines]),
        voltage_angle=np.array([pyo.value(model.theta[b]) for b in model.Buses]),
        energy_price=np.array(
            [get_price(model, model.balance[b]) for b in model.Buses]
        ),
        congestion_price=np.array(
            [
                get_price(model, model.flow_bounds_upper[ell])
                - get_price(model, model.flow_bounds_lower[ell])
                for ell in model.Lines
            ]
        ),
    )
