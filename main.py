from typing import Any
import pandas as pd
import numpy as np


# Load the CSV file into a DataFrame
_df = pd.read_csv("lagerstatus.csv")


class DeliveryVan:
    def __init__(self):
        """Create a new delivery van."""
    
    def load_package():
        """Load a package into this van."""
        raise NotImplementedError()


_delivery_vans = (DeliveryVan() for _ in range(10))
        

def get_delivery_van(package) -> Any:
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
    selected_row = _df[_df["Paket_id"] == 2472920751]
    print(type(selected_row))
    print(selected_row)


if __name__ == "__main__":
    main()