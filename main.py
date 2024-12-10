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


class DeliveryVan:
    def __init__(self, name):
        """Create a new delivery van."""
        self._loaded_weight = 0
        self._packages: list[Package] = []
        self._name = name

    def load_package(self, id_, weight, profit, deadline):
        """Load a package into this van."""

        if self._loaded_weight + weight >= 800:
            raise Exception("Too many packages in this van.")
        
        self._packages.append( Package(id_, weight, profit, deadline) )
        self._loaded_weight += weight


def decide_delivery_van(package: Package) -> DeliveryVan:
    """ Use an algorithm to decide which van this package should be contained in. """
    raise NotImplementedError()


def _get_total_delivery_weight() -> float:
    """Get the summed weight of all packages to be delivered"""
    assert __name__ == "__main__", "Don't import this function."
    return _df["Vikt"].sum()


def main():
    if not os.path.isfile("lagerstatus.csv"):
        import seeder
        seeder.seed_packages(10000)

    global _df
    _df = pd.read_csv("lagerstatus.csv") # Dataframe

    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)]

    print( _get_total_delivery_weight() )

    # selected_row = df[df["Paket_id"] == 2472920751]
    # print(type(selected_row))
    # print(selected_row)


if __name__ == "__main__":
    main()