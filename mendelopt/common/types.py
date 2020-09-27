"""
Implements common types for the genetic algorithm procedure.
"""

from math import ceil, floor
from typing import Callable, List, NamedTuple

import numpy as np  # type: ignore

FitnessFunc = Callable[[np.ndarray], float]


class GeneticParams:
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
            raise ValueError("Elite rate plus immigrant rate should be at most 1, preferably lower.")

        self.population_size = population_size
        self.evolution_steps = evolution_steps

        # Compute how the counts for each of the population segments: the elite, the
        # mutations and the newborns.

        self.elite_count = max(1, floor(elite_rate * population_size))
        self.immigrant_count = max(1, floor(immigrant_rate * population_size))

        leftover_count = population_size - self.elite_count - self.immigrant_count

        self.mutation_count = floor(leftover_count * crossover_rate)
        self.newborn_count = ceil(leftover_count * (1 - crossover_rate))


class Individual:
    """
    Represents a single individual.
    """

    _array: np.ndarray
    _fitness: float

    def __init__(self, array: np.ndarray, fitness_function: Callable[[np.ndarray], float]):
        self._array = array
        self._fitness = fitness_function(array)

    def as_array(self):
        """
        Returns the array representation of the individual.
        """

        return self._array

    def fitness(self):
        """
        Returns the fitness of the individual.
        """

        return self._fitness

    def __repr__(self) -> str:
        return f"Individual: (w/ fitness of {self._fitness})\n{self._array}"


class GeneticResult(NamedTuple):
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
