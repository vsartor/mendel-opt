# mendel-opt

Python library written as a case study in making a generalized pythonic implementation of a Genetic Algorithms, which is a heuristic optimization algorithm.

The algorithm is based on user-defined _"strategies"_ for individual generation, mutation and reproduction, with some basic defaults already implemented. 

## Optimization

The genetic algorithm is accessible through the [`mendel.optimizer.optimize`](https://github.com/vsartor/mendel/blob/master/mendel/optimizer.py). This function takes in a fitness function, three strategies and optionally some Genetic Algorithm hyper-parameters.

## Fitness Function

The fitness function is the objective function for the optimizer. This function is expected to be of the type `Callable[[np.ndarray], float]`, meaning it takes a NumPy array and returns a floating point value. An alias for this function type is given in [`mendel.common.types.FitnessFunc`](https://github.com/vsartor/mendel/blob/master/mendel/common/types.py).

## Generation Strategy

The algorithm needs to be able to generate new individuals from scratch. The abstract class in [`mendel.generation.GenerationStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/generation.py) is given as a template for user-defined strategies.

An example of an implementation is [`mendel.generation.BasicGenerationStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/generation.py), which takes in a shape and a domain of values in the constructor, and randomly builds NumPy arrays with random choices of these values to generate new individuals.

## Mutation Strategy

The algorithm needs to be able to mutate existing individuals. The abstract class in [`mendel.mutation.MutationStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/mutation.py) is given as a template for user-defined strategies.

An example of an implementation is [`mendel.mutation.BasicMutationStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/mutation.py), which takes a domain of values in the constructor, and randomly alters an entry from the existing individual to another random value from the domain.

## Reproduction Strategy

The algorithm needs to be able to create new individuals based on the information from two existing individuals. The abstract class in [`mendel.reproduction.ReproductionStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/reproduction.py) is given as a template for user-defined strategies.

An example of an implementation is [`mendel.reproduction.BasicReproductionStrategy`](https://github.com/vsartor/mendel/blob/master/mendel/reproduction.py), which takes no arguments in the constructor, and effectly picks a random index and slices the two parents to create the offspring.

## Genetic Parameters

To tune the hyperparameters one may instantiate the named tuple in [`mendel.common.types.GeneticParams`](https://github.com/vsartor/mendel/blob/master/mendel/common/types.py).

Particularly relevant parameters are:

* `evolution_steps`, which is an integer value determining how many iterations the algorithm should run;
* `population_size`, which is an integer value determining how many individuals the algorithm keeps track at each iteration.

## Warm starts

The [`mendel.optimizer.optimize`](https://github.com/vsartor/mendel/blob/master/mendel/optimizer.py) function also can take a `warm_start` parameter to continue a previous optimization call that already returned. In particular, it expects to receive the `.population` attribute of the return object from the previous [`mendel.optimizer.optimize`](https://github.com/vsartor/mendel/blob/master/mendel/optimizer.py) call.
