if __name__ != "__main__": raise ImportError("This module cannot be imported.")


from typing import Any
import pandas as pd
import numpy as np


class DeliveryVan:
    def __init__(self, name):
        """Create a new delivery van."""
        self._loaded_weight = 0
        self._name = name

    def get_loaded_weight(self):
        """Get how many kilograms worth of packages are loaded into the van."""
        return self._loaded_weight

    def load_package():
        """Load a package into this van."""
        raise NotImplementedError()
    
    def __repr__(self):
        return self._name
        

def decide_delivery_van(package) -> Any:
    """
    Determines which van should haul the given package.

    Parameters
    ----------
    package : Any
        The package to check.

    Returns
    -------
    Any
        The van the package should be hauled with.
    """
    raise NotImplementedError()


def main():
    df = pd.read_csv("lagerstatus.csv") # Dataframe
    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)]

    print(delivery_vans)

    # selected_row = df[df["Paket_id"] == 2472920751]
    # print(type(selected_row))
    # print(selected_row)


if __name__ == "__main__":
    main()