import pandas as pd
import numpy as np
import seeder, random
from tqdm import tqdm


TOTAL_PACKAGES = 10_000


seeder.seed_packages(TOTAL_PACKAGES)


df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})


class DeliveryVan:
    def __init__(self, name: str) -> None:
        """Create a new delivery van."""

        self._name = name
        self._loaded_weight = np.float64()
        self._profit = 0

        self.is_overloaded = False

    def load_package(self, package: pd.Series) -> None:
        """
        Load a package into this van. The total weight cannot exceed 800 Kg.

        ### Args:
        - `package`: A row from the `lagerstatus.csv` file containing package details.
        """

        self._loaded_weight = round(self._loaded_weight + package["Vikt"], 1)
        self._profit += get_real_profit(package)

    def get_weight(self) -> np.float64:
        return self._loaded_weight

    def get_profit(self) -> int:
        return self._profit

    def __repr__(self) -> str:
        return self._name+f" [{self._loaded_weight} Kg]"
    

delivery_vans = [DeliveryVan(f"bil_{i+1}") for i in range(10)] # 10 vans


def get_real_profit(package: dict) -> int:
    """
    Get the real profit with deadline taken into account for the given package.
    ### Args:
    - `package`: A row from the `lagerstatus.csv` file containing package details.
    """
    deadline = package["Deadline"]
    return package["Förtjänst"] - (deadline**2 if deadline < 0 else 0)


def genetic_algorithm(size: int):
    """
    Perform a genetic algorithm to find out the optimal way to load packages into 10 vans in order to get the highest profit.
    ### Args:
    - `size` - The size of the chromosome population. 
    """

    # Not finished
    raise NotImplementedError()

    def initialize_population() -> list:
        """
        Create an initial population of chromosomes.
        ### Returns:
        - `chromosome`- A list representing each package as an integer that acts as the index for the van it should be put into,
        where -1 means don't package it into a van
        """
        return [np.random.randint(-1, 10, size=TOTAL_PACKAGES).tolist() for _ in range(size)]

    def get_fitness(chromosome: list) -> int:
        """
        Get the fitness of this chromosome.
        ### Args:
        - `chromosome`- A list representing a chromosome.
        ### Returns:
        - `fitness` An integer describing the fitness.
        """

        number_of_overloaded_vans = 0
        package_array = df.to_numpy()
        column_map = {name: idx for idx, name in enumerate(df.columns)}

        for list_index, van_index in enumerate(chromosome):
            if van_index != -1:
                van: DeliveryVan = delivery_vans[van_index]
                package = package_array[list_index]
                van.load_package({
                    "Vikt": package[column_map["Vikt"]],
                    "Förtjänst": package[column_map["Förtjänst"]],
                    "Deadline": package[column_map["Deadline"]],
                })

                if not van.is_overloaded and van.get_weight() > 800:
                    number_of_overloaded_vans += 1
                    van.is_overloaded = True

        total_profit = sum(van.get_profit() for van in delivery_vans) # The profit by all vans with the deadlines taken into account
        overload_penelty = 1_000_000*number_of_overloaded_vans
        fitness = total_profit - overload_penelty

        return fitness

    def select_parents(population: list, fitness_scores: list) -> tuple:
        """
        Select two parents based on fitness scores using roulette wheel selection.
        ### Args:
        - `population` - A list of chromosomes representing a population.
        ### Returns:
        - `parent1, parent2` - The two most fittest parent chromosomes.
        """
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        parents = np.random.choice(population, size=2, p=probabilities)
        return parents[0], parents[1]

    def crossover(parent1: list, parent2: list) -> list:
        """
        Perform single-point crossover to create offspring.
        ### Args:
        - `parent1`- The first parent chromosome.
        - `parent2`- The second parent chromosome.
        ### Returns:
        - `child`- The newly created child with random features of both parents (single-point crossover).
        """
        crossover_point = random.randint(0, len(parent1) - 1)
        return parent1[:crossover_point] + parent2[crossover_point:]

    def mutate(chromosome: list, mutation_rate: float) -> list:
        """Mutate the chromosome with a given mutation rate."""
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(-1, 9)  # Randomly assign a new van or -1
        return chromosome
    
    population = initialize_population()
    best_fitness = None

    # Start algorithm
    # As many generations as there are initial chromosomes
    for generation in tqdm( range(size), total=size, desc=f"Performing genetic algorithm..." ): 
        fitness_scores = [get_fitness(chromosome) for chromosome in population]
        new_population = []

        for _ in range(size // 2):  # Create offspring pairs
            parent1, parent2 = select_parents(population, fitness_scores) # Select parents

            # Create two children from the parents and mutate them slightly
            child1 = mutate( crossover(parent1, parent2 ), size)
            child2 = mutate( crossover(parent2, parent1 ), size)

            # Add to population
            new_population.extend([child1, child2])

        population = new_population
        best_fitness = max(fitness_scores)
    
    # Return the best chromosome
    best_index = fitness_scores.index( best_fitness )
    return population[best_index], best_fitness


def main() -> None:
    best_solution, best_fitness = genetic_algorithm(4)
    print("Best Solution Found:")
    print(best_solution)
    print("Best fitness:", best_fitness)


if __name__ == "__main__":
    main()