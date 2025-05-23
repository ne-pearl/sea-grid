import numpy as np
from numpy.typing import NDArray
from pandas import DataFrame
from pjm5bus_pandas import buses, generators, lines, offers, reference_bus


def line_bus(buses: DataFrame, lines: DataFrame) -> NDArray[np.int8]:
    """Line-bus incidence matrix."""

    def incidence(ell: int, b: int) -> int:
        if buses.at[b, "id"] == lines.at[ell, "from_bus_id"]:
            return -1
        if buses.at[b, "id"] == lines.at[ell, "to_bus_id"]:
            return +1
        return 0

    return np.array(
        [[incidence(ell, b) for b in buses.index] for ell in lines.index],
        dtype=np.int8,
    )


def offer_bus(
    offers: DataFrame,
    buses: DataFrame,
    generators: DataFrame,
) -> NDArray[np.bool]:
    """Offer-bus incidence matrix."""

    def incidence(o: int, b: int, g: int) -> bool:
        bus_generator: bool = buses.at[b, "id"] == generators.at[g, "bus_id"]
        generator_offer: bool = generators.at[g, "id"] == offers.at[o, "generator_id"]
        return bus_generator and generator_offer

    return np.array(
        [
            [any(incidence(o, b, g) for g in generators.index) for b in buses.index]
            for o in offers.index
        ],
        dtype=np.bool,
    )


def reference_bus(buses: DataFrame, id_: str) -> int:
    ids = buses["id"]
    matches = ids[ids == id_]
    assert len(matches) == 1
    return matches.index[0]
