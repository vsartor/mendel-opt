"""
Python implementation of a Genetic Algorithm in a functional manner.
"""

import random
from math import ceil, floor
from typing import List, NamedTuple, Optional

import numpy as np  # type: ignore

from mendel.common.types import FitnessFunc, Individual
from mendel.generation import GenerationStrategy
from mendel.mutation import MutationStrategy
from mendel.reproduction import ReproductionStrategy


class AlgorithmParams:
    """
    Parameters for the Genetic Algorithm.
    """

    population_size: int
    evolution_steps: int
    newborn_mutation_rate: float

    elite_count: int
    mutation_count: int
    newborn_count: int

    def __init__(
        self,
        population_size: int = 300,
        evolution_steps: int = 100,
        elite_rate: float = 0.05,
        immigrant_rate: float = 0.05,
        crossover_rate: float = 0.65,
    ):
        """
        :param population_size: Determines how many different individuals will be
        considered at each step of evolution. Defaults to 300.

        :param evolution_steps: How many evolution steps should be performed before
        returning the results. Defaults to 100.

        :param elite_rate: A value between 0.0 and 1.0 indicating how many of the
        fittest individuals will be carried over to the next generation as
        a proportion of population_size. Defaults to 0.05.

        :param immigrant_rate: A value between 0.0 and 1.0 indicating how many
        spots of the population should be filled with completely new individuals
        at each evolution steps. Defaults to 0.05.

        :param crossover_rate: A value between 0.0 and 1.0 indicating the proportion
        of the leftover population spots after both the elite individuals and immigrants
        enter the population will be filled with newborn individuals as opposed to simply
        mutated individuals. Defaults to 0.65.
        """

        if population_size < 1:
            raise ValueError("Population size should be at least 1.")
        if evolution_steps < 1:
            raise ValueError("Evolution should contain at least 1 step.")
        if elite_rate < 0 or elite_rate > 1:
            raise ValueError("Elite rate should be a value between 0 and 1.")
        if immigrant_rate < 0 or immigrant_rate > 1:
            raise ValueError("Immigrant rate should be a value between 0 and 1.")
        if crossover_rate < 0 or crossover_rate > 1:
            raise ValueError("Crossover rate should be a value between 0 and 1.")
        if elite_rate + immigrant_rate > 1:
            raise ValueError(
                "Elite rate plus immigrant rate should be at most 1, preferably lower."
            )

        self.population_size = population_size
        self.evolution_steps = evolution_steps

        # Compute how the counts for each of the population segments: the elite, the
        # mutations and the newborns.

        self.elite_count = max(1, floor(elite_rate * population_size))
        self.immigrant_count = max(1, floor(immigrant_rate * population_size))

        leftover_count = population_size - self.elite_count - self.immigrant_count

        self.mutation_count = floor(leftover_count * crossover_rate)
        self.newborn_count = ceil(leftover_count * (1 - crossover_rate))


class OptimizationResult(NamedTuple):
    """
    Holds the result for a Genetic Optimization procedure.

    Attributes:
        :best: Fittest individual.
        :history: An array with evolution_steps (see GeneticParams) entries
        holding the fitness of the fittest individual for each of the evolution
        steps.
        :population: Contains the entire population at the time of return.
        This can be passed to a further optimization call as a warm start to
        continue the optimization from where it stopped.
    """

    best: Individual
    history: np.ndarray
    population: List[Individual]


def optimize(
    fitness: FitnessFunc,
    generation_strategy: GenerationStrategy,
    mutation_strategy: MutationStrategy,
    reproduction_strategy: ReproductionStrategy,
    params: AlgorithmParams = AlgorithmParams(),
    warm_start: Optional[List[Individual]] = None,
) -> OptimizationResult:
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

        immigrants = generation_strategy.generate_population(
            params.immigrant_count, fitness
        )

        mutation_origins = _select_candidates(population, params.mutation_count)
        mutations = [
            mutation_strategy.mutate(individual, fitness)
            for individual in mutation_origins
        ]

        parents_a = _select_candidates(population, params.newborn_count)
        parents_b = _select_candidates(population, params.newborn_count)
        newborns = [
            reproduction_strategy.reproduce(parents, fitness)
            for parents in zip(parents_a, parents_b)
        ]

        new_population = elite
        new_population.extend(immigrants)
        new_population.extend(mutations)
        new_population.extend(newborns)

        population = sorted(
            new_population, key=lambda individual: -individual.fitness()
        )
        fitness_history[iteration_count] = population[0].fitness()

    return OptimizationResult(
        best=population[0],
        history=fitness_history,
        population=population,
    )


def _select_candidates(population: List[Individual], count: int) -> List[Individual]:
    """
    Select candidates for an evolution step based on fitness.
    """

    return random.choices(
        population, weights=[individual.fitness() for individual in population], k=count
    )
