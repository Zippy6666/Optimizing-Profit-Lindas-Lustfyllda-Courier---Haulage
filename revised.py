""" Skriven av vår käre vän ~ """

# import pandas as pd
# import numpy as np
# import random
# from tqdm import tqdm

# _TOTAL_PACKAGES = 1_360_000
# _MAX_VAN_WEIGHT = 800.0

# # Generate the `lagerstatus.csv` data (if needed)
# # For testing, ensure `lagerstatus.csv` exists with appropriate columns: Paket_id, Vikt, Förtjänst, Deadline
# _df = pd.read_csv("lagerstatus.csv", dtype={"Paket_id": str, "Vikt": float, "Förtjänst": int, "Deadline": int})

# class DeliveryVan:
#     def __init__(self, name: str) -> None:
#         self.name = name
#         self.loaded_weight = 0.0
#         self.profit = 0
#         self.overloaded = False

#     def load_package(self, package: dict) -> None:
#         self.loaded_weight += package["Vikt"]
#         if self.loaded_weight > _MAX_VAN_WEIGHT:
#             self.overloaded = True
#         self.profit += self.calculate_profit(package)

#     def calculate_profit(self, package: dict) -> int:
#         penalty = (package["Deadline"] ** 2) if package["Deadline"] < 0 else 0
#         return package["Förtjänst"] - penalty

#     def reset(self):
#         self.loaded_weight = 0.0
#         self.profit = 0
#         self.overloaded = False

#     def __repr__(self) -> str:
#         return f"{self.name} [{self.loaded_weight} Kg, Profit: {self.profit}]"

# # Initialize vans
# _delivery_vans = [DeliveryVan(f"Van_{i+1}") for i in range(10)]

# def genetic_algorithm(pop_size: int, generations: int, mutation_rate: float):
#     def initialize_population():
#         return [np.random.randint(-1, len(_delivery_vans), size=len(_df)).tolist() for _ in range(pop_size)]

#     def evaluate_fitness(chromosome):
#         for van in _delivery_vans:
#             van.reset()

#         for package_idx, van_idx in enumerate(chromosome):
#             if van_idx != -1:  # Skip unassigned packages
#                 van = _delivery_vans[van_idx]
#                 package = _df.iloc[package_idx]
#                 van.load_package(package)

#         total_profit = sum(van.profit for van in _delivery_vans)
#         overload_penalty = sum(1_000_000 for van in _delivery_vans if van.overloaded)
#         return total_profit - overload_penalty

#     def select_parents(population, fitness_scores):
#         total_fitness = sum(fitness_scores)
#         probabilities = [score / total_fitness for score in fitness_scores]
#         return random.choices(population, weights=probabilities, k=2)

#     def crossover(parent1, parent2):
#         point = random.randint(0, len(parent1) - 1)
#         return parent1[:point] + parent2[point:]

#     def mutate(chromosome):
#         for i in range(len(chromosome)):
#             if random.random() < mutation_rate:
#                 chromosome[i] = random.randint(-1, len(_delivery_vans) - 1)
#         return chromosome

#     population = initialize_population()
#     best_fitness = float("-inf")
#     best_solution = None

#     for generation in tqdm(range(generations), desc="Running Genetic Algorithm"):
#         fitness_scores = [evaluate_fitness(chromosome) for chromosome in population]

#         if max(fitness_scores) > best_fitness:
#             best_fitness = max(fitness_scores)
#             best_solution = population[fitness_scores.index(best_fitness)]

#         new_population = []
#         for _ in range(pop_size // 2):
#             parent1, parent2 = select_parents(population, fitness_scores)
#             child1 = mutate(crossover(parent1, parent2))
#             child2 = mutate(crossover(parent2, parent1))
#             new_population.extend([child1, child2])

#         population = new_population

#     return best_solution, best_fitness

# def main():
#     pop_size = 10
#     generations = 100
#     mutation_rate = 0.02

#     best_solution, best_fitness = genetic_algorithm(pop_size, generations, mutation_rate)
#     print("Best solution found:")
#     print(best_solution)
#     print("Best fitness:", best_fitness)

# if __name__ == "__main__":
#     main()
