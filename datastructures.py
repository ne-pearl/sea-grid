import dataclasses
import textwrap
from typing import Any, Callable, Generic, Optional, Self, TypeVar
import numpy as np
from numpy.typing import NDArray
import polars as pl

# Units of measure
Id = pl.String
MW = pl.Float64
MWPerRad = pl.Float64
USDPerMWh = pl.Float64

buses_input_schema = {"id": Id}
generators_input_schema = {"id": Id, "bus_id": Id}
lines_input_schema = {
    "from_bus_id": Id,
    "to_bus_id": Id,
    "capacity": MW,
    "susceptance": MWPerRad,
}
offers_input_schema = {
    "generator_id": Id,
    "max_quantity": MW,
    "price": USDPerMWh,
}


def validate(dataframe: pl.DataFrame, schema: dict) -> bool:
    """Validate the schema of a DataFrame."""
    return schema.items() <= dataframe.schema.items()


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
        reference_bus: str,
    ) -> Self:
        """Initialize from dataframes."""

        assert validate(buses, buses_input_schema)
        assert validate(generators, generators_input_schema)
        assert validate(lines, lines_input_schema)
        assert validate(offers, offers_input_schema)

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

        reference_bus_index = buses["id"].index_of(reference_bus)
        assert reference_bus_index is not None

        return cls(
            bus_load=demand_loads @ -load_bus_incidence,
            line_capacity=lines[:, "capacity"].to_numpy(),
            line_susceptance=lines[:, "susceptance"].to_numpy(),
            offer_max_quantity=offers[:, "max_quantity"].to_numpy(),
            offer_price=offers[:, "price"].to_numpy(),
            line_bus_incidence=line_bus_incidence,
            offer_bus_incidence=offer_bus_incidence,
            reference_bus=reference_bus_index,
        )


Model = TypeVar("Model")


@dataclasses.dataclass(frozen=True, repr=False, slots=True)
class Result(Generic[Model], Serialization):
    model: Model
    total_cost: float
    dispatch_quantity: NDArray[np.float64]
    line_flow: NDArray[np.float64]
    voltage_angle: NDArray[np.float64]
    energy_price: Optional[NDArray[np.float64]] = None
    congestion_price: Optional[NDArray[np.float64]] = None
    loss_price: Optional[NDArray[np.float64]] = None


def enumerate_over(df: pl.DataFrame, over: str, alias: str = "id") -> pl.DataFrame:
    """Creates a new key by"""

    # Generate a temporary column name that doesn't conflict with existing columns
    longest: str = max(df.columns, key=len, default="")
    temporary: str = longest + "_temporary"

    # Add a temporary column that enumerates rows within each group
    df = df.with_columns(pl.arange(0, pl.len()).over(over).alias(temporary))

    # Create the new identifier by combining the group key and the enumeration
    df = df.with_columns(
        pl.format("{}/{}", pl.col(over), pl.col(temporary) + 1).alias(alias)
    )

    return df.drop(temporary)
