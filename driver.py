import polars as pl
from datastructures import Data, Result
from formulations import formulate

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
result: Result = formulate(net)
