import pandas as pd
import numpy as np
import seeder
from tqdm import tqdm


class DeliveryVan:
    def __init__(self, name: str) -> None:
        """Create a new delivery van."""

        self._name = name
        self._loaded_weight = np.float64()
        self._profit = 0

    def load_package(self, package: pd.Series) -> None:
        """
        Load a package into this van. The total weight cannot exceed 800 Kg.

        ### Args:
        - `package`: A row from a `lagerstatus.csv` file containing package details.
        """

        self._loaded_weight = round(self._loaded_weight + package["Vikt"], 1)
        self._profit += get_real_profit(package)

    def empty(self) -> None:
        self._loaded_weight = np.float64()
        self._profit = 0

    def get_weight(self) -> np.float64:
        return self._loaded_weight

    def get_profit(self) -> int:
        return self._profit

    def __repr__(self) -> str:
        return self._name+f" [{self._loaded_weight} Kg]"


def get_real_profit(package: dict) -> int:
    """
    Get the real profit with deadline taken into account for the given package.

    ### Args:
    - `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    deadline = package["Deadline"]
    return package["Förtjänst"] - (deadline**2 if deadline < 0 else 0)


def sort_dataframe(dataframe: pd.DataFrame,) -> pd.DataFrame:
    """
    Sorts a package dataframe using a greedy algorithm based on priority factors: weight, profit, and deadline status.
    
    ### Args:
    - `dataframe`: The dataframe containing package information with columns such as weight, profit, and deadline.
    
    ### Returns:
    - `sorted_dataframe`: A new dataframe sorted to prioritize packages based on factors.
    """

    # Vectorized calculation of 'Sort_Worth' using pandas and numpy
    sort_worth = (dataframe["Förtjänst"] / dataframe["Vikt"]) - np.minimum(0, dataframe["Deadline"])**2
    
    # Add 'Sort_Worth' column
    dataframe["Sort_Worth"] = sort_worth
    
    # Sort by 'Sort_Worth' in descending order and drop the temporary 'Sort_Worth' column
    sorted_dataframe = dataframe.sort_values(by="Sort_Worth", ascending=False).drop(columns=["Sort_Worth"])
    
    return sorted_dataframe


def fill_vans(vans: list[DeliveryVan], data: pd.DataFrame) -> int:
    van_idx = 0
    for _, row in data.iterrows():
        van = vans[van_idx]

        if van.get_weight() + row["Vikt"] <= 800:
            van.load_package(row)
        else:
            van_idx+=1

            # Last van, break out
            if van_idx >= len(vans):
                break

            van = vans[van_idx]
            van.load_package(row)

    profit_sum = sum(van.get_profit() for van in vans)
    return profit_sum


def main() -> None:
    seeder.seed_packages(1_360_000)
    df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})
    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans

    profit_sum = fill_vans( delivery_vans, sort_dataframe(df) )

    print(profit_sum)
    

if __name__ == "__main__":
    main()