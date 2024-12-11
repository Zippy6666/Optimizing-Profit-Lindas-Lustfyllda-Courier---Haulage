import pandas as pd
import numpy as np
import os


class _DeliveryVan:
    def __init__(self, name):
        """Create a new delivery van."""
        
        assert __name__ == "__main__", "Don't import this class."

        self._name = name
        self._loaded_weight = 0.0

        dtypes = {
            "Paket_id": str,
            "Vikt": float,
            "Förtjänst": int,
            "Deadline": int
        }
        self._packages = pd.DataFrame(columns=["Paket_id", "Vikt", "Förtjänst", "Deadline"]).astype(dtypes) # Empty DataFrame

    def load_package(self, package: pd.Series):
        """
        Load a package into this van. The total weight cannot exceed 800 Kg.

        ### Args:
        - `package`: A row from the `lagerstatus.csv` file containing package details.
        """

        weight = package["Vikt"]

        assert self._loaded_weight + weight < 800, "Too many packages in this van."

        self._loaded_weight = round(self._loaded_weight + weight, 1)
        self._packages = pd.concat([self._packages, package.to_frame().T], ignore_index=True) # Add package to van

        global _df
        _df = _df[_df["Paket_id"] != package["Paket_id"]]

    def get_weight(self) -> np.float64:
        return self._loaded_weight

    def get_profit(self) -> int:
        return self._packages["Förtjänst"].sum()

    def __repr__(self):
        return self._name+f" [{self._loaded_weight} Kg]"


def _decide_delivery_van(package: pd.Series) -> _DeliveryVan:
    """
    Use an algorithm to decide which van this package should be contained in.

    ### Args:
    - `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    assert __name__ == "__main__", "Don't import this function."
    raise NotImplementedError()


def main():
    # Seed new data if there isn't any
    if not os.path.isfile("lagerstatus.csv"):
        import seeder
        seeder.seed_packages(1_360_000)

    global _df
    _df = pd.read_csv("lagerstatus.csv", dtype={
        "Paket_id": str,
        "Vikt": float,
        "Förtjänst": int,
        "Deadline": int
    })

    global _delivery_vans
    _delivery_vans = [_DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans


if __name__ == "__main__":
    main()