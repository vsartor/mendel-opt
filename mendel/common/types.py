"""
Implements common types for the genetic algorithm procedure.
"""

from typing import Callable

import numpy as np  # type: ignore

FitnessFunc = Callable[[np.ndarray], float]


class Individual:
    """
    Represents a single individual.
    """

    _array: np.ndarray
    _fitness: float

    def __init__(
        self, array: np.ndarray, fitness_function: Callable[[np.ndarray], float]
    ):
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
