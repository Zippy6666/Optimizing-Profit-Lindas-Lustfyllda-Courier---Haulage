""" Skriven av vår käre vän ~ """

import pandas as pd
import numpy as np
import random
from tqdm import tqdm

def genetic_algorithm(vans: list, data: pd.DataFrame, pop_size: int, generations: int, mutation_rate: float):
    def initialize_population():
        return [... for _ in range(pop_size)]

    def evaluate_fitness(chromosome):
        for van in vans:
            van.reset()

        for package_idx, van_idx in enumerate(chromosome):
            if van_idx != -1:  # Skip unassigned packages
                van = vans[van_idx]
                package = data.iloc[package_idx]
                van.load_package(package)

        total_profit = sum(van.profit for van in vans)
        overload_penalty = sum(1_000_000 for van in vans if van.overloaded)
        return total_profit - overload_penalty

    def select_parents(population, fitness_scores):
        total_fitness = sum(fitness_scores)
        probabilities = [score / total_fitness for score in fitness_scores]
        return random.choices(population, weights=probabilities, k=2)

    def crossover(parent1, parent2):
        point = random.randint(0, len(parent1) - 1)
        return parent1[:point] + parent2[point:]

    def mutate(chromosome):
        for i in range(len(chromosome)):
            if random.random() < mutation_rate:
                chromosome[i] = random.randint(-1, len(vans) - 1)
        return chromosome

    population = initialize_population()
    best_fitness = float("-inf")
    best_solution = None

    for generation in tqdm(range(generations), desc="Running Genetic Algorithm"):
        fitness_scores = [evaluate_fitness(chromosome) for chromosome in population]

        if max(fitness_scores) > best_fitness:
            best_fitness = max(fitness_scores)
            best_solution = population[fitness_scores.index(best_fitness)]

        new_population = []
        for _ in range(pop_size // 2):
            parent1, parent2 = select_parents(population, fitness_scores)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])

        population = new_population

    return best_solution, best_fitness

def main():
    pop_size = 10
    generations = 100
    mutation_rate = 0.02

    best_solution, best_fitness = genetic_algorithm(pop_size, generations, mutation_rate)
    print("Best solution found:")
    print(best_solution)
    print("Best fitness:", best_fitness)

if __name__ == "__main__":
    main()
