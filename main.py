import pandas as pd
import numpy as np
import seeder, csv
from tqdm import tqdm
from pathlib import Path


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


def sort_dataframe(dataframe: pd.DataFrame, profit_importance_mult: np.float64) -> pd.DataFrame:
    """
    Sorts a package dataframe using a greedy algorithm based on priority factors: weight, profit, and deadline status.
    
    ### Args:
    - `dataframe`: The dataframe containing package information with columns such as weight, profit, and deadline.
    
    ### Returns:
    - `sorted_dataframe`: A new dataframe sorted to prioritize packages based on factors.
    """

    sort_worth = ( ((dataframe["Förtjänst"]*profit_importance_mult) / dataframe["Vikt"])
    - np.minimum(0, dataframe["Deadline"])**2 )
    
    dataframe["Sort_Worth"] = sort_worth

    # Sort by 'Sort_Worth' in descending order and drop the temporary 'Sort_Worth' column
    sorted_dataframe = dataframe.sort_values(by="Sort_Worth", ascending=False).drop(columns=["Sort_Worth"])
    
    return sorted_dataframe


def fill_vans(vans: list[DeliveryVan], data: pd.DataFrame) -> int:
    """
    TBD
    """

    clear_vans(vans)

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


def clear_vans(vans: list[DeliveryVan]) -> None:
    """
    TBD
    """

    for van in vans:
        van.empty()


def learn_packaging(
        vans: list[DeliveryVan],
        data: pd.DataFrame,
        n_profit_mults: int,
        max_profit_mult: np.float64,
        N_PACKAGES: int,
    ) -> np.float64:
    """
    TBD
    """

    best_score = 0
    profit_mults = np.linspace(0.0, max_profit_mult, num=n_profit_mults)

    for profit_mult in tqdm( profit_mults, total=len(profit_mults), desc="Finding ideal multiplier..." ):
        score = fill_vans(vans, sort_dataframe( data, profit_mult ))

        if score > best_score:
            best_score = score
            best_profit_mult = profit_mult

    assert "best_profit_mult" in locals(), "Did not find best multiplier. This should not happen."

    append_prof_mult_to_file(best_profit_mult, N_PACKAGES)


def append_prof_mult_to_file(prof_mult: np.float64, n_packages: int) -> None:
    """
    TBD
    """

    target_path = Path(f"profit_importance_mults_{n_packages}_packages.csv")

    fieldnames = ("profit_importance_mult",)
    
    # Write the header if the file is empty
    if not target_path.exists() or target_path.stat().st_size == 0:
        with target_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    with target_path.open("a", encoding="utf-8", newline="") as f:  # Open in append mode
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writerow({"profit_importance_mult": prof_mult})


def get_optimized_profit_mult(n_packages: int) -> np.float64 | None:
    """
    TBD
    """

    target_path = Path(f"profit_importance_mults_{n_packages}_packages.csv")
    if target_path.exists():
        profit_mults_df = pd.read_csv(target_path)
        return np.float64( profit_mults_df["profit_importance_mult"].mean() )


def main() -> None:
    """
    The goal is to find the most optimized "profit importance multiplier".
    It does so by finding the best possible solution for each run of package and applying the mean
    of their "profit importance multiplier" to the next run.
    
    This multiplier seems to vary between packages. For instance, 10 000 packages seems to converge to a mean of about 3.36.
    """

    SEARCH_STEPS = 32
    MAX_PROFIT_MULT = np.float64(4.0)
    N_PACKAGES = 10_000 

    seeder.seed_packages(N_PACKAGES)
    df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})
    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans

    profit_unoptimized = fill_vans(delivery_vans, sort_dataframe(df, np.float64(1.0)))
    
    print("Profit, unoptimized:", profit_unoptimized)

    optimized_mult = get_optimized_profit_mult(N_PACKAGES)
    print("Mean profit importance mult:", optimized_mult)
    if optimized_mult:
        profit_optimized = fill_vans(delivery_vans, sort_dataframe(df, optimized_mult))
        print("Profit, optimized:", profit_optimized)
        print("Profit gain:", profit_optimized-profit_unoptimized)

    learn_packaging(delivery_vans, df, SEARCH_STEPS, MAX_PROFIT_MULT, N_PACKAGES)


if __name__ == "__main__":
    main()