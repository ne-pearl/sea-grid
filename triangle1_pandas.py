from pandas import DataFrame

base_power = 100  # [MVA]
reference_bus = "Bus1"
buses = DataFrame(
    [
        ("Bus1", 0.0, 0.0, 1.0),
        ("Bus2", 0.0, 2.0, 1.0),
        ("Bus3", 150.0, 1.0, 0.0),
    ],
    columns=["id", "load", "x", "y"],
)
generators = DataFrame(
    [
        ("G1", "Bus1"),
        ("G2", "Bus2"),
    ],
    columns=["id", "bus_id"],
)
lines = DataFrame(
    [
        ("Bus1", "Bus2", 150.0, 0.25),
        ("Bus1", "Bus3", 150.0, 0.25),
        ("Bus2", "Bus3", 150.0, 0.25),
    ],
    columns=["from_bus_id", "to_bus_id", "capacity", "reactance"],
)
offers = DataFrame(
    [
        ("G1", 200.0, 10.0),
        ("G2", 200.0, 12.0),
    ],
    columns=["generator_id", "quantity", "price"],
)
