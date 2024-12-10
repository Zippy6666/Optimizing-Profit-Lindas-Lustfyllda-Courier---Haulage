import pandas as pd
import numpy as np
import os, random


class _DeliveryVan:
    def __init__(self, name):
        """Create a new delivery van."""
        
        assert __name__ == "__main__", "Don't import this class."

        self._loaded_weight = 0.0
        self._packages: np.array = np.array([], dtype=object)
        self._name = name

    def load_package(self, package: pd.Series):
        """
        Load a package into this van. The total weight cannot exceed 800 Kg.

        ### Args:
        - `package`: A row from the `lagerstatus.csv` file containing package details.
        """

        weight = package["Vikt"]

        assert self._loaded_weight + weight < 800, "Too many packages in this van."

        self._loaded_weight = round(self._loaded_weight + weight, 1)
        self._packages = np.append(self._packages, package)
    
    def __repr__(self):
        return self._name+f" [{self._loaded_weight} Kg]"


def _decide_delivery_van(package: pd.Series) -> _DeliveryVan:
    """
    Use an algorithm to decide which van this package should be contained in.

    ### Args:
    - `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    assert __name__ == "__main__", "Don't import this function."
    
    # Stupid algorithm that just puts it in a random van:
    van = random.choice(_delivery_vans)
    return van


def main():
    if not os.path.isfile("lagerstatus.csv"):
        import seeder
        seeder.seed_packages(5000)

    global _df
    global _delivery_vans

    _df = pd.read_csv("lagerstatus.csv") # Dataframe
    _delivery_vans = [_DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans

    for _, package in _df.iterrows():
        van = _decide_delivery_van(package)
        van.load_package(package)
        print("Loaded package", package["Paket_id"], "into", van, ".")


if __name__ == "__main__":
    main()