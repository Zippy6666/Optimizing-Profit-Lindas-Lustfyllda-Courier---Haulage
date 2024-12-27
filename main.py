import pandas as pd
import numpy as np
import seeder, csv
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
    
    dataframe["Sort_Worth"] = sort_worth

    # Sort by 'Sort_Worth' in descending order and drop the temporary 'Sort_Worth' column
    sorted_dataframe = dataframe.sort_values(by="Sort_Worth", ascending=False).drop(columns=["Sort_Worth"])
    
    return sorted_dataframe


def fill_vans(vans: list[DeliveryVan], data: pd.DataFrame) -> int:
    """
    Fill the vans with packages one by one until they cannot carry any more weight.

    ### Args:
    `vans`: The delivery vans.
    `data`: The packages.

    ### Returns:
    `profit`: How much profit the packages in all the vans are collectively worth.
    """

    clear_vans(vans)

    van_idx = 0
    for _, row in data.iterrows():
        van = vans[van_idx]

        if van.get_weight() + row["Vikt"] <= van.get_max_weight():
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
        score = fill_vans(vans, sort_dataframe( data, profit_mult ))

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
    Determine if the algorithm has learned the optimal "profit importance multiplier" for `n_packages` packages.

    ### Args:
    ``
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

    # Compute the differences between consecutive means
    differences = [abs(recent_means[i] - recent_means[i - 1]) for i in range(1, len(recent_means))]

    # Check if all differences are at most stop_max_mean_change
    return all(diff <= stop_max_mean_change for diff in differences)


def package_vans() -> dict:
    """
    Packages the vans with new packages and tries to learn how to do so optimally.

    The goal is to find the most optimized "profit importance multiplier".
    It does so by finding the best possible solution for each run of package and applying the mean
    of their "profit importance multiplier" to the next run.

    This multiplier seems to vary between the total number of packages.
    For instance, 10 000 packages seems to converge to a mean of about 3.29.
    """

    # Constants
    SEARCH_STEPS = 32
    MAX_PROFIT_MULT = np.float64(4.0)
    N_PACKAGES = 500_000
    STOP_MAX_MEAN_CHANGE = 0.01
    STOP_AFTER = 10 # The mean did not change by more than 'STOP_MAX_MEAN_CHANGE' for this amount of times, so stop learning.

    # Seed new packages
    seeder.seed_packages(N_PACKAGES)
    df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})

    # Make 10 delivery vans
    delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)]

    # Try filling them with brute force, without optimizing
    profit_unoptimized = fill_vans(delivery_vans, sort_dataframe(df, np.float64(1.0)))

    mean: np.float64 | None = get_profit_mult_mean(N_PACKAGES) # Get the mean of all the past best 'profit importance multipliers'
    gain = 0 # How much profit is gained after optimization

    if mean:
        profit_optimized = fill_vans(delivery_vans, sort_dataframe(df, mean))
        gain = profit_optimized-profit_unoptimized # Calc profit gain

    # Determine if we should keep trying to optimize the algorithm
    done_learning = get_should_stop(N_PACKAGES, STOP_MAX_MEAN_CHANGE, STOP_AFTER)

    # Keep learning if we should
    if not done_learning:
        # Determine what the best 'profit importance multiplier' would have been for this group of packages
        best_score = gridsearch_best_score_for_these_packages(delivery_vans, df, SEARCH_STEPS, MAX_PROFIT_MULT)
        remember_in_file(best_score, N_PACKAGES)

    # Return results
    return {"done_learning": done_learning, "profit_importance_multiplier": mean, "profit_gain": gain}


if __name__ == "__main__":
    done_learning = False
    while not done_learning:
        result = package_vans()
        done_learning = result["done_learning"]
        print(result)