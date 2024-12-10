from typing import Any
import pandas as pd
import numpy as np
import os


class Package:
    def __init__(self, id_, weight, profit, deadline):
        self._id = id_
        self._weight = weight
        self._profit = profit
        self._deadline = deadline


class _DeliveryVan:
    def __init__(self, name):
        """Create a new delivery van."""
        assert __name__ == "__main__", "Don't import this class."

        self._loaded_weight = 0
        self._packages: list[Package] = []
        self._name = name

    def load_package(self, id_):
        """Load a package into this van. The total weight of all the van's packages cannot exceed 800 Kg."""

        assert self._loaded_weight + weight < 800, "Too many packages in this van."

        selected_row = _df[_df["Paket_id"] == id_]
        weight, profit, deadline = selected_row["Vikt"], selected_row["Förtjänst"], selected_row["Deadline"]

        self._packages.append( Package(id_, weight, profit, deadline) )
        self._loaded_weight += weight


def _decide_delivery_van(package: Package) -> _DeliveryVan:
    """ Use an algorithm to decide which van this package should be contained in. """
    raise NotImplementedError()


def _get_total_delivery_weight() -> float:
    """Get the summed weight of all packages to be delivered"""
    assert __name__ == "__main__", "Don't import this function."
    return _df["Vikt"].sum()


def main():
    if not os.path.isfile("lagerstatus.csv"):
        import seeder
        seeder.seed_packages(5000)

    global _df
    global _delivery_vans

    _df = pd.read_csv("lagerstatus.csv") # Dataframe
    _delivery_vans = [_DeliveryVan(f"bil_{i+1}") for i in range(10)]

    print( _get_total_delivery_weight() )

if __name__ == "__main__":
    main()