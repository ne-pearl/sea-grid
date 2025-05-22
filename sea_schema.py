import polars as pl

Id = pl.String
Distance = pl.Float64
MW = pl.Float64
PU = pl.Float64
USDPerMWh = pl.Float64

buses_schema = {"id": Id, "load": MW, "x": Distance, "y": Distance}
generators_schema = {"id": Id, "bus_id": Id}
lines_schema = {"from_bus_id": Id, "to_bus_id": Id, "capacity": MW, "reactance": PU}
offers_schema = {"generator_id": Id, "quantity": MW, "price": USDPerMWh}


def validate(dataframe: pl.DataFrame, schema: dict) -> bool:
    """Validate the schema of a DataFrame."""
    return schema.items() <= dataframe.schema.items()
