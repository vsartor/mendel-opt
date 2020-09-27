"""
Python implementation of a Genetic Algorithm in a functional manner.
"""

import random
from typing import List, Optional

import numpy as np  # type: ignore

from mendelopt.common.types import FitnessFunc, GeneticParams, GeneticResult, Individual
from mendelopt.generation import GenerationStrategy
from mendelopt.mutation import MutationStrategy
from mendelopt.reproduction import ReproductionStrategy


def optimize(
    fitness: FitnessFunc,
    generation_strategy: GenerationStrategy,
    mutation_strategy: MutationStrategy,
    reproduction_strategy: ReproductionStrategy,
    params: GeneticParams = GeneticParams(),
    warm_start: Optional[List[Individual]] = None,
) -> GeneticResult:
    """
    Performs genetic optimization on a function that expected a NumPy array
    and returns a floating point value indicating that array's fitness.

    See GeneticParams for further details on the specific optimization parameters.

    Each evolution step is implemented as follows:
        1. The fittest individuals are carried over as per elite_rate.
        2. A proportion of individuals given by (1 - elite_rate) * (1 - crossover_rate),
           rounded up, will be generated as mutations from fit individuals¹.
        3. A proportion of individuals given by (1 - elite_rate) * crossover_rate,
           rounded down, will be generated as crossovers from fit individuals¹.

    ¹The individuals selected for the mutation and crossover steps are
     randomly selected with probability given by their relative fitness
     compared to the current population.
    """

    fitness_history = np.empty(params.evolution_steps)

    population = sorted(
        warm_start
        if warm_start is not None
        else generation_strategy.generate_population(params.population_size, fitness),
        key=lambda individual: -individual.fitness(),
    )

    for iteration_count in range(params.evolution_steps):
        elite = population[: params.elite_count]

        immigrants = generation_strategy.generate_population(params.immigrant_count, fitness)

        mutation_origins = _select_candidates(population, params.mutation_count)
        mutations = [mutation_strategy.mutate(individual, fitness) for individual in mutation_origins]

        parents_a = _select_candidates(population, params.newborn_count)
        parents_b = _select_candidates(population, params.newborn_count)
        newborns = [reproduction_strategy.reproduce(parents, fitness) for parents in zip(parents_a, parents_b)]

        new_population = elite
        new_population.extend(immigrants)
        new_population.extend(mutations)
        new_population.extend(newborns)

        population = sorted(new_population, key=lambda individual: -individual.fitness())
        fitness_history[iteration_count] = population[0].fitness()

    return GeneticResult(
        best=population[0],
        history=fitness_history,
        population=population,
    )


def _select_candidates(population: List[Individual], count: int) -> List[Individual]:
    """
    Select candidates for an evolution step based on fitness.
    """

    return random.choices(population, weights=[individual.fitness() for individual in population], k=count)
