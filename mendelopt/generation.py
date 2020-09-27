"""
Implements abstract and basic individual generation strategies.
"""

import random
from abc import ABC, abstractmethod
from functools import reduce
from typing import List, Set, Tuple

import numpy as np  # type: ignore

from mendelopt.common.types import FitnessFunc, Individual
from mendelopt.common.utils import censor_string


class GenerationStrategy(ABC):
    """
    Abstract class defining abstract methods for an individual generation
    strategy.
    """

    @abstractmethod
    def generate_individual(self, fitness_function: FitnessFunc) -> Individual:
        """
        Function to generate an individual.
        """

    def generate_population(self, count: int, fitness_function: FitnessFunc) -> List[Individual]:
        """
        Function to generate a population of individuals.
        """

        return [self.generate_individual(fitness_function) for _ in range(count)]


class BasicGenerationStrategy(GenerationStrategy):
    """
    Basic generation strategy based on an array shape and integer domain
    values.
    """

    _element_count: int
    _shape: Tuple[int, ...]
    _domain: List[int]

    def __init__(self, shape: Tuple[int, ...], values: Set[int]):
        self._element_count = reduce(lambda x, y: x * y, shape)
        self._shape = shape
        self._domain = list(values)

    def generate_individual(self, fitness_function: FitnessFunc) -> Individual:
        """
        Generates a single random individual.
        """

        values = random.choices(self._domain, k=self._element_count)
        individual_array = np.reshape(values, self._shape)
        return Individual(individual_array, fitness_function)

    def __repr__(self) -> str:
        shape_str = censor_string(str(self._shape))
        domain_str = censor_string(str(self._domain))
        return f"BasicGenerationStrategy(shape {shape_str} over {domain_str})"
