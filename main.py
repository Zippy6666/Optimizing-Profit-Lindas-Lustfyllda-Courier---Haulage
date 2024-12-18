import pandas as pd
import numpy as np
import seeder
import random
from tqdm import tqdm


class _DeliveryVan:
    def __init__(self, name) -> None:
        """Create a new delivery van."""
        
        assert __name__ == "__main__", "Don't import this class."

        self._name = name
        self._loaded_weight = 0.0
        self._profit = 0

        self.is_overloaded = False

    def load_package(self, package: pd.Series) -> None:
        """
        Load a package into this van. The total weight cannot exceed 800 Kg.

        ### Args:
        - `package`: A row from the `lagerstatus.csv` file containing package details.
        """

        self._loaded_weight = round(self._loaded_weight + package["Vikt"], 1)
        self._profit += _get_real_profit(package)

    def get_weight(self) -> np.float64:
        return self._loaded_weight

    def get_profit(self) -> int:
        return self._profit

    def __repr__(self) -> str:
        return self._name+f" [{self._loaded_weight} Kg]"
    

def _get_real_profit(package: dict) -> int:
    """
    Get the real profit with deadline taken into account for the given package.
    ### Args:
    - `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    deadline = package["Deadline"]
    return package["Förtjänst"] - (deadline**2 if deadline < 0 else 0)


def _get_fitness(chromosome: list) -> int:
    """
    Get the fitness of this chromosome.
    ### Args:
    - `chromosome`- A list representing each package as an integer that acts as the index for the van it should be put into, where -1 means don't package it into a van
    ### Returns:
    - `fitness` An integer describing the fitness.
    """

    number_of_overloaded_vans = 0
    package_array = _df.to_numpy()
    column_map = {name: idx for idx, name in enumerate(_df.columns)}

    for list_index, van_index in tqdm(enumerate(chromosome), total=len(chromosome), desc="Processing"):
        if van_index != -1:
            van: _DeliveryVan = _delivery_vans[van_index]
            package = package_array[list_index]
            van.load_package({
                "Vikt": package[column_map["Vikt"]],
                "Förtjänst": package[column_map["Förtjänst"]],
                "Deadline": package[column_map["Deadline"]],
            })

            if not van.is_overloaded and van.get_weight() > 800:
                number_of_overloaded_vans += 1
                van.is_overloaded = True

    total_profit = sum(van.get_profit() for van in _delivery_vans) # The profit by all vans with the deadlines taken into account
    overload_penelty = 1_000_000*number_of_overloaded_vans
    fitness = total_profit - overload_penelty

    return fitness


def main() -> None:
    print("Seeding new packages...")

    # Seed new data
    seeder.seed_packages(1_360_000)

    print("Seeding done!")

    global _df
    global _delivery_vans

    _df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})
    _delivery_vans = [_DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans

    chromosome = [random.randint(-1, 9) for _ in range(len(_df))]
    print("Fitness:", _get_fitness(chromosome))

if __name__ == "__main__":
    main()