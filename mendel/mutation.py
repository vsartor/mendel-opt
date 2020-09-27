"""
Implements abstract and basic individual mutation strategies.
"""

import random
from abc import ABC, abstractmethod
from typing import List, Set

from mendel.common.types import FitnessFunc, Individual
from mendel.common.utils import censor_string


class MutationStrategy(ABC):
    """
    Abstract class defining abstract methods for an individual mutation strategy.
    """

    @abstractmethod
    def mutate(self, individual: Individual, fitness_function: FitnessFunc) -> Individual:
        """
        Function to mutate an individual.
        """


class BasicMutationStrategy(MutationStrategy):
    """
    Basic mutation strategy based on randomly altering a single entry.
    """

    _domain: List[int]

    def __init__(self, values: Set[int]):
        self._domain = list(values)

    def mutate(self, individual: Individual, fitness_function: FitnessFunc) -> Individual:
        """
        Mutates an individual by switching one of its entries to another value.
        """

        mutation_index = tuple(random.randrange(0, dim_size) for dim_size in individual.as_array().shape)

        new_array = individual.as_array().copy()
        new_array[mutation_index] = random.sample(self._domain, k=1)[0]
        return Individual(new_array, fitness_function)

    def __repr__(self) -> str:
        domain_str = censor_string(str(self._domain))
        return f"BasicMutationStrategy(over {domain_str})"
