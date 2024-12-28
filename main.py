from typing import Literal
import pandas as pd
import numpy as np
import seeder, csv
import sys
from tqdm import tqdm
from pathlib import Path


class DeliveryVan:
    def __init__(self, name: str, max_weight: np.float64 = np.float64(800.0)) -> None:
        """Create a new delivery van."""

        self._name = name
        self._loaded_weight = np.float64()
        self._profit = 0
        self._max_weight = max_weight

    def load_package(self, package: pd.Series) -> None:
        """
        Load a package into this van.

        ### Args:
        `package`: A row from a `lagerstatus.csv` file containing package details.
        """

        self._loaded_weight = round(self._loaded_weight + package["Vikt"], 1)
        self._profit += get_real_profit(package)

    def empty(self) -> None:
        """
        Empty this van.
        """
        self._loaded_weight = np.float64()
        self._profit = 0

    def get_weight(self) -> np.float64:
        return self._loaded_weight

    def get_max_weight(self) -> np.float64:
        return self._max_weight
    
    def get_profit(self) -> int:
        return self._profit

    def __repr__(self) -> str:
        """
        Nice representation of the van. For debug purposes.
        """
        return self._name+f" [{self._loaded_weight} Kg]"


def get_real_profit(package: dict) -> int:
    """
    Get the real profit with deadline taken into account for the given package.

    ### Args:
    `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    deadline = package["Deadline"]
    return package["Förtjänst"] - (deadline**2 if deadline < 0 else 0)


def sort_dataframe(dataframe: pd.DataFrame, profit_importance_mult: np.float64) -> pd.DataFrame:
    """
    Determine a sorting priority for each package as such:\n
    `( ( profit * profit_importance_mult ) / weight ) - deadline_overdue ^ 2`

    Sort all packages in the dataframe based on this formula.
    
    ### Args:
    `dataframe`: The dataframe containing package information with the columns weight, profit, and deadline.
    
    ### Returns:
    `sorted_dataframe`: A new dataframe sorted to prioritize packages based on factors.
    """

    sort_worth = ( ((dataframe["Förtjänst"]*profit_importance_mult) / dataframe["Vikt"])
    - np.minimum(0, dataframe["Deadline"])**2 )

    sorted_dataframe = dataframe.copy()
    
    sorted_dataframe["Sort_Worth"] = sort_worth

    # Sort by 'Sort_Worth' in descending order and drop the temporary 'Sort_Worth' column
    sorted_dataframe = sorted_dataframe.sort_values(by="Sort_Worth", ascending=False).drop(columns=["Sort_Worth"])

    return sorted_dataframe


def get_total_penalty_of_undelivered_packages(dataframe: pd.DataFrame) -> np.int64:
    """
    Calculate the total penalty for undelivered packages based on their deadlines.

    ### Args:
    `dataframe`: The DataFrame containing package data.

    ### Returns:
    `total_penalty`: The sum of penalties for all undelivered packages.
    """

    undelivered_with_penalty_df = dataframe[ (dataframe["Delivered"] == False) & (dataframe["Deadline"] < 0) ]

    total_penalty = (undelivered_with_penalty_df["Deadline"] ** 2).sum()
    
    return total_penalty


def fill_vans(vans: list[DeliveryVan], data: pd.DataFrame, fake: bool = False) -> int:
    """
    Fill the vans with packages one by one until they cannot carry any more weight.

    ### Args:
    `vans`: The delivery vans.
    `data`: The packages.
    `fake`: If true, packages won't actually be delivered.

    ### Returns:
    `profit`: How much profit the packages in all the vans are collectively worth.
    """

    clear_vans(vans)

    van_idx = 0
    for i, row in data.iterrows():
        van = vans[van_idx]

        if van.get_weight() + row["Vikt"] <= van.get_max_weight():

            van.load_package(row)
            data.at[i, "Delivered"] = not fake

        else:
            van_idx+=1

            # Last van, break out
            if van_idx >= len(vans):
                break

            van = vans[van_idx]
            van.load_package(row)
            data.at[i, "Delivered"] = not fake

    profit_sum = sum(van.get_profit() for van in vans)
    return profit_sum


def clear_vans(vans: list[DeliveryVan]) -> None:
    """
    Remove all the packages from the supplied delivery vans.

    ### Args:
    `vans`: The delivery vans.
    """
    for van in vans:
        van.empty()


def gridsearch_best_score_for_these_packages(
        vans: list[DeliveryVan],
        data: pd.DataFrame,
        n_profit_mults: int,
        max_profit_mult: np.float64,
    ) -> np.float64:
    """
    Determine the best "profit importance multiplier" for these packages doing a gridsearch from 0 to `max_profit_mult`.

    ### Args:
    `vans`: The vans to test on.
    `data`: The packages to test on.
    `n_profit_mults`: The number of "profit importance multipliers" to test.
    `max_profit_mult`: The largest "profit importance multiplier" to test.

    ### Returns:
    `best_profit_mult`: The determined best 'profit_importance_multiplier' for these vans and packages.
    """

    best_score = 0
    profit_mults = np.linspace(0.0, max_profit_mult, num=n_profit_mults)

    desc = f"Finding best multiplier for these {len(data)} packages"

    for profit_mult in tqdm( profit_mults, total=len(profit_mults), desc=desc, leave=False ):
        score = fill_vans(vans, sort_dataframe( data, profit_mult ), fake=True)

        if score > best_score:
            best_score = score
            best_profit_mult = profit_mult

    assert "best_profit_mult" in locals(), "Did not find best multiplier. This should not happen."

    return best_profit_mult


def remember_in_file(prof_mult: np.float64, n_packages: int) -> None:
    """
    Remember the "profit importance multiplier" for a run of packages.
    Also store the mean of past "profit importance multipliers", including the supplied one.
    This mean can be used to predict future optimal solutions.

    ### Args:
    `prof_mult`: The "profit importance multiplier" to remember.
    `n_packages`: The amount of packages to remember this multiplier for.
    This is important since the ideal multiplier depends on the number of packages.
    """

    target_path = Path(f"profit_importance_mults_{n_packages}_packages.csv")

    fieldnames = ("profit_importance_mult", "profit_importance_mult_mean")

    # Read existing values to compute the mean
    existing_values = []
    if target_path.exists() and target_path.stat().st_size > 0:
        with target_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                existing_values.append(float(row["profit_importance_mult"]))

    # Include the current value
    all_values = existing_values + [prof_mult]
    current_mean = np.mean(all_values)

    # Write the header if the file is empty
    if not target_path.exists() or target_path.stat().st_size == 0:
        with target_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
            writer.writeheader()

    # Append the new row
    with target_path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator='\n')
        writer.writerow({
            "profit_importance_mult": prof_mult,
            "profit_importance_mult_mean": current_mean
        })


def get_profit_mult_mean(n_packages: int) -> np.float64 | None:
    """
    Get the mean "profit importance multiplier" for this number of packages.

    ### Args:
    `n_packages`: The amount of packages to get the optimal multiplier for.
    This is important since the ideal multiplier depends on the number of packages.

    ### Returns:
    `mean`- The mean it can be determined, otherwise None.
    """

    target_path = Path(f"profit_importance_mults_{n_packages}_packages.csv")

    # No csv file so return nothing
    if not target_path.exists():
        return

    # Get the latest up-to-date mean from the csv
    df = pd.read_csv(target_path)
    return df.iloc[-1, 1]


def get_should_stop(n_packages: int, stop_max_mean_change: int, stop_after: int) -> bool:
    """
    Determine if the algorithm has learned the optimal "profit importance multiplier" for the packaging of `n_packages` packages.

    ### Args:
    `n_packages`: Check if we have learned packaging for this amount of packages.
    `stop_max_mean_change`: If the mean did not change by more than this `stop_after` amount of times, determine that we are done learning.
    `stop_after`: If the mean did not change by more than `stop_max_mean_change` this amount of times, determine that we are done learning.

    ### Returns:
    `should_stop`: Whether or not we are done learning.
    """

    target_path = Path(f"profit_importance_mults_{n_packages}_packages.csv")

    if not target_path.exists():
        return False

    df = pd.read_csv(target_path)
    
    means = list( df["profit_importance_mult_mean"] )

    # Ensure we have enough means to compare
    if len(means) < stop_after:
        return False

    # Get the most recent stop_after_n_changes means
    recent_means = means[-stop_after:]

    most_recent_mean = recent_means[-1]

    # Compute the differences between means
    differences = [abs(recent_mean - most_recent_mean) for recent_mean in recent_means]

    # Check if all differences are at most stop_max_mean_change
    return all(diff <= stop_max_mean_change for diff in differences)


def package_vans() -> dict:
    """
    Packages 10 vans with a limit of 800 Kg with x new packages and tries to learn how to do so optimally.

    This is what it does step by step:
    1. Receive new, never seen before packages.
    2. Sort the packages in order of profit, weight, and how long the deadline is overdue.
    Assume that we determine priority by profit / weight.
    3. Fill all the vans with packages in order of priority as determined in step 2.
    4. Optimize the formula if there were any previous packagings, use the mean formula of all the previous packagings.
    5. If the formula was optimized, let us know how much profit we gained compared to the raw, unoptimized formula.
    6. Do a grid search to decide what the best formula for these packages would have been.
    7. Calculate and remember the mean of the best formula for these packages and any potential previous packagings that took place.
    8. If a large amount of previously calculated means are near this mean, assume we have found the best formula, so stop learning.

    The only thing that ever changes in the formula is a simple multiplier that
    increases or decreases how important the profit is compared to how light the package is.

    ### Returns:
    `results` Information in the form of a dictionary as follows:
        {
            "done_learning": bool,
            "new_best_profit_importance_multiplier": np.float64,
            "profit": int,
            "profit_gain": int
            "penalty_of_undelivered_packages": np.int64
        }
    Where for some context, `profit` is the total profit from all the packages in the vans,
    and `profit_gain` is the amount of gained profit after the formula was optimized.
    and `penalty_of_undelivered_packages` is the total penalty of all packages remaining in the warehouse.
    """

    # Number of packages
    arg_n: int | Literal[False] = len(sys.argv)>1 and sys.argv[1].isdigit() and int(sys.argv[1])
    N_PACKAGES = arg_n or 10_000

    # Constants for the algorithm
    SEARCH_STEPS = 30
    MAX_PROFIT_MULT = np.float64(8.0)
    STOP_MAX_MEAN_CHANGE = 0.02
    STOP_AFTER = 10 # The mean did not change by more than 'STOP_MAX_MEAN_CHANGE' for this amount of times, so stop learning.

    # Seed new packages
    seeder.seed_packages(N_PACKAGES)
    df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})

    # Add a column for checking if this package has been packaged in a van
    df["Delivered"] = False

    # Make 10 delivery vans
    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)]

    mean: np.float64 | None = get_profit_mult_mean(N_PACKAGES) # Get the mean of all the past best 'profit importance multipliers', if any
    gain = 0 # How much profit is gained after potential optimization

    # Sort dataframe
    df = sort_dataframe(df, np.float64(1.0))

    profit_unoptimized = fill_vans(delivery_vans, df, fake=( mean is None and False or True ))
    profit_optimized = None

    if mean:
        # Sort dataframe, but try doing it in an optimized way
        df = sort_dataframe(df, mean)

        profit_optimized = fill_vans(delivery_vans, df)
        gain = profit_optimized-profit_unoptimized # Calc profit gain

    # Determine if we should keep trying to optimize the algorithm
    done_learning = get_should_stop(N_PACKAGES, STOP_MAX_MEAN_CHANGE, STOP_AFTER)

    # Keep learning if we should
    if not done_learning:
        # Determine what the best 'profit importance multiplier' would have been for this group of packages
        best_score = gridsearch_best_score_for_these_packages(delivery_vans, df, SEARCH_STEPS, MAX_PROFIT_MULT)
        remember_in_file(best_score, N_PACKAGES)

    # Return results
    return {
        "done_learning": done_learning,
        "new_best_profit_importance_multiplier": mean,
        "profit": profit_optimized or profit_unoptimized,
        "profit_gain": gain,
        "penalty_of_undelivered_packages": get_total_penalty_of_undelivered_packages(df)
    }


if __name__ == "__main__":
    _done_learning = False

    while not _done_learning:
        _result = package_vans()
        _done_learning = _result["done_learning"]

        print(_result)