from pandas import DataFrame

base_power = 100  # [MVA]
reference_bus = "A"

buses = DataFrame(
    [
        ("A", 0.0, 0.0, 0.0),
        ("B", 300.0, 1.0, 0.0),
        ("C", 300.0, 2.0, 0.0),
        ("D", 400.0, 2.0, 1.0),
        ("E", 0.0, 0.0, 2.0),
    ],
    columns=["id", "load", "x", "y"],
)

generators = DataFrame(
    [
        ("Alta", "A"),
        ("ParkCity", "A"),
        ("Solitude", "C"),
        ("Sundance", "D"),
        ("Brighton", "E"),
    ],
    columns=["id", "bus_id"],
)


lines = DataFrame(
    [
        ("A", "B", 400.0, 2.81),
        ("A", "D", 1000.0, 3.04),
        ("A", "E", 1000.0, 0.64),
        ("B", "C", 1000.0, 1.08),
        ("C", "D", 1000.0, 2.97),
        ("D", "E", 240.0, 2.97),
    ],
    columns=["from_bus_id", "to_bus_id", "capacity", "reactance"],
)


offers = DataFrame(
    [
        ("Alta", 40.0, 14.0),
        ("ParkCity", 170.0, 15.0),
        ("Solitude", 520.0, 30.0),
        ("Sundance", 200.0, 40.0),
        ("Brighton", 600.0, 10.0),
        ("Alta", 40.0, 140.0),
        ("ParkCity", 170.0, 150.0),
        ("Solitude", 520.0, 300.0),
        ("Sundance", 200.0, 400.0),
        ("Brighton", 600.0, 100.0),
    ],
    columns=["generator_id", "quantity", "price"],
)
