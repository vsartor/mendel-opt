"""
Implements abstract and basic individual reproduction strategies.
"""

import random
from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np  # type: ignore

from mendel.common.types import FitnessFunc, Individual


class ReproductionStrategy(ABC):
    """
    Abstract class defining abstract methods for reproduction strategy.
    """

    @abstractmethod
    def reproduce(
        self, parents: Tuple[Individual, Individual], fitness_function: FitnessFunc
    ) -> Individual:
        """
        Function to create a newborn from a couple of individuals.
        """


class BasicReproductionStrategy(ReproductionStrategy):
    """
    Basic reproduction strategy based on slicing.
    """

    def reproduce(
        self, parents: Tuple[Individual, Individual], fitness_function: FitnessFunc
    ) -> Individual:
        """
        Creates a new individual based on slicing the parents.
        """

        new_array = np.empty(parents[0].as_array().size)

        slice_index = random.randint(0, new_array.size)
        if slice_index == 0:
            new_array[:] = parents[0].as_array().reshape(new_array.size)
        elif slice_index == new_array.size:
            new_array[:] = parents[1].as_array().reshape(new_array.size)
        else:
            new_array[:slice_index] = (
                parents[0].as_array().reshape(new_array.size)[:slice_index]
            )
            new_array[slice_index:] = (
                parents[1].as_array().reshape(new_array.size)[slice_index:]
            )

        return Individual(
            new_array.reshape(parents[0].as_array().shape), fitness_function
        )

    def __repr__(self) -> str:
        return "BasicReproductionStrategy()"
