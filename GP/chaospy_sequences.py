"""
Generate Chebyshev pseudo-random samples.

Example usage
-------------

Basic usage::

    >>> print(create_chebyshev_samples(order=1))
    [[ 0.5]]
    >>> print(create_chebyshev_samples(order=2))
    [[ 0.25  0.75]]
    >>> print(create_chebyshev_samples(order=5))
    [[ 0.0669873  0.25       0.5        0.75       0.9330127]]

Certain orders are nested::

    >>> print(create_chebyshev_samples(order=3))
    [[ 0.14644661  0.5         0.85355339]]
    >>> print(create_chebyshev_samples(order=7))
    [[ 0.03806023  0.14644661  0.30865828  0.5         0.69134172  0.85355339
       0.96193977]]

Create nested samples directly with the dedicated function::

    >>> print(create_nested_chebyshev_samples(order=1))
    [[ 0.5]]
    >>> print(create_nested_chebyshev_samples(order=2))
    [[ 0.14644661  0.5         0.85355339]]
    >>> print(create_nested_chebyshev_samples(order=3))
    [[ 0.03806023  0.14644661  0.30865828  0.5         0.69134172  0.85355339
       0.96193977]]

Multivariate usage::

    >>> print(create_chebyshev_samples(order=2, dim=2))
    [[ 0.25  0.25  0.75  0.75]
     [ 0.25  0.75  0.25  0.75]]
"""
import numpy

#import chaospy.quad


def create_chebyshev_samples(order, dim=1):
    """
    Chebyshev sampling function.

    Args:
        order (int):
            The number of samples to create along each axis.
        dim (int):
            The number of dimensions to create samples for.

    Returns:
        samples following Chebyshev sampling scheme mapped to the
        ``[0, 1]^dim`` hyper-cube and ``shape == (dim, order)``.
    """
    x_data = .5*numpy.cos(numpy.arange(order, 0, -1)*numpy.pi/(order+1)) + .5
    x_data = chaospy.quad.combine([x_data]*dim)
    return x_data.T


def create_nested_chebyshev_samples(order, dim=1):
    """
    Nested Chebyshev sampling function.

    Args:
        order (int):
            The number of samples to create along each axis.
        dim (int):
            The number of dimensions to create samples for.

    Returns:
        samples following Chebyshev sampling scheme mapped to the
        ``[0, 1]^dim`` hyper-cube and ``shape == (dim, 2**order-1)``.
    """
    return create_chebyshev_samples(order=2**order-1, dim=dim)
"""
Generate samples from a regular grid.

Example usage
-------------

Basic usage::

    >>> print(create_grid_samples(order=1))
    [[ 0.5]]
    >>> print(create_grid_samples(order=2))
    [[ 0.33333333  0.66666667]]
    >>> print(create_grid_samples(order=5))
    [[ 0.16666667  0.33333333  0.5         0.66666667  0.83333333]]

Certain orders are nested::

    >>> print(create_grid_samples(order=3))
    [[ 0.25  0.5   0.75]]
    >>> print(create_grid_samples(order=7))
    [[ 0.125  0.25   0.375  0.5    0.625  0.75   0.875]]

Create nested samples directly with the dedicated function::

    >>> print(create_nested_grid_samples(order=1))
    [[ 0.5]]
    >>> print(create_nested_grid_samples(order=2))
    [[ 0.25  0.5   0.75]]
    >>> print(create_nested_grid_samples(order=3))
    [[ 0.125  0.25   0.375  0.5    0.625  0.75   0.875]]

Multivariate usage::

    >>> print(create_grid_samples(order=2, dim=2))
    [[ 0.33333333  0.33333333  0.66666667  0.66666667]
     [ 0.33333333  0.66666667  0.33333333  0.66666667]]
"""
import numpy

#import chaospy.quad


def create_grid_samples(order, dim=1):
    """
    Create samples from a regular grid.

    Args:
        order (int):
            The order of the grid. Defines the number of samples.
        dim (int):
            The number of dimensions in the grid

    Returns (numpy.ndarray):
        Regular grid with ``shape == (dim, order)``.
    """
    x_data = numpy.arange(1, order+1)/(order+1.)
    x_data = chaospy.quad.combine([x_data]*dim)
    return x_data.T


def create_nested_grid_samples(order, dim=1):
    """
    Create samples from a nested grid.

    Args:
        order (int):
            The order of the grid. Defines the number of samples.
        dim (int):
            The number of dimensions in the grid

    Returns (numpy.ndarray):
        Regular grid with ``shape == (dim, 2**order-1)``.
    """
    return create_grid_samples(order=2**order-1, dim=dim)
"""
Create samples from the `Halton sequence`_.

In statistics, Halton sequences are sequences used to generate points in space
for numerical methods such as Monte Carlo simulations. Although these sequences
are deterministic, they are of low discrepancy, that is, appear to be random
for many purposes. They were first introduced in 1960 and are an example of
a quasi-random number sequence. They generalise the one-dimensional van der
Corput sequences.

Example usage
-------------

Standard usage::

    >>> print(create_halton_samples(order=3, dim=2))
    [[ 0.125       0.625       0.375     ]
     [ 0.44444444  0.77777778  0.22222222]]
    >>> print(create_halton_samples(order=3, dim=3))
    [[ 0.375       0.875       0.0625    ]
     [ 0.22222222  0.55555556  0.88888889]
     [ 0.24        0.44        0.64      ]]

Custom burn-ins::

    >>> print(create_halton_samples(order=3, dim=2, burnin=0))
    [[ 0.5         0.25        0.75      ]
     [ 0.33333333  0.66666667  0.11111111]]
    >>> print(create_halton_samples(order=3, dim=2, burnin=1))
    [[ 0.25        0.75        0.125     ]
     [ 0.66666667  0.11111111  0.44444444]]
    >>> print(create_halton_samples(order=3, dim=2, burnin=2))
    [[ 0.75        0.125       0.625     ]
     [ 0.11111111  0.44444444  0.77777778]]

Using custom prime bases::

    >>> print(create_halton_samples(order=3, dim=2, primes=[7, 5]))
    [[ 0.16326531  0.30612245  0.44897959]
     [ 0.64        0.84        0.08      ]]
    >>> print(create_halton_samples(order=3, dim=3, primes=[5, 3, 7]))
    [[ 0.64        0.84        0.08      ]
     [ 0.88888889  0.03703704  0.37037037]
     [ 0.16326531  0.30612245  0.44897959]]

.. Halton sequence: https://en.wikipedia.org/wiki/Halton_sequence
"""
import numpy



def create_halton_samples(order, dim=1, burnin=None, primes=None):
    """
    Create Halton sequence.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Halton sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Halton sequence.
        burnin (int, optional):
            Skip the first ``burnin`` samples. If omitted, the maximum of
            ``primes`` is used.
        primes (array_like, optional):
            The (non-)prime base to calculate values along each axis. If
            omitted, growing prime values starting from 2 will be used.

    Returns (numpy.ndarray):
        Halton sequence with ``shape == (dim, order)``.
    """
    if primes is None:
        primes = []
        prime_order = 10*dim
        while len(primes) < dim:
            primes = create_primes(prime_order)
            prime_order *= 2
    primes = primes[:dim]
    assert len(primes) == dim, "not enough primes"

    if burnin is None:
        burnin = max(primes)

    out = numpy.empty((dim, order))
    indices = [idx+burnin for idx in range(order)]
    for dim_ in range(dim):
        out[dim_] = create_van_der_corput_samples(
            indices, number_base=primes[dim_])
    return out
"""
Create samples from the `Hammersley set`_.

The Hammersley set is equivalent to the Halton sequence, except for one
dimension is replaced with a regular grid.

Example usage
-------------

Standard usage::

    >>> print(create_hammersley_samples(order=3, dim=2))
    [[ 0.75   0.125  0.625]
     [ 0.25   0.5    0.75 ]]
    >>> print(create_hammersley_samples(order=3, dim=3))
    [[ 0.125       0.625       0.375     ]
     [ 0.44444444  0.77777778  0.22222222]
     [ 0.25        0.5         0.75      ]]

Custom burn-ins::

    >>> print(create_hammersley_samples(order=3, dim=3, burnin=0))
    [[ 0.5         0.25        0.75      ]
     [ 0.33333333  0.66666667  0.11111111]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, burnin=1))
    [[ 0.25        0.75        0.125     ]
     [ 0.66666667  0.11111111  0.44444444]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, burnin=2))
    [[ 0.75        0.125       0.625     ]
     [ 0.11111111  0.44444444  0.77777778]
     [ 0.25        0.5         0.75      ]]

Using custom prime bases::

    >>> print(create_hammersley_samples(order=3, dim=2, primes=[7]))
    [[ 0.16326531  0.30612245  0.44897959]
     [ 0.25        0.5         0.75      ]]
    >>> print(create_hammersley_samples(order=3, dim=3, primes=[7, 5]))
    [[ 0.16326531  0.30612245  0.44897959]
     [ 0.64        0.84        0.08      ]
     [ 0.25        0.5         0.75      ]]

.. Hammersley set: https://en.wikipedia.org/wiki/Low-discrepancy_sequence#Hammersley_set
"""
import numpy



def create_hammersley_samples(order, dim=1, burnin=None, primes=None):
    """
    Create samples from the Hammersley set.

    For ``dim == 1`` the sequence falls back to Van Der Corput sequence.

    Args:
        order (int):
            The order of the Hammersley sequence. Defines the number of samples.
        dim (int):
            The number of dimensions in the Hammersley sequence.
        burnin (int, optional):
            Skip the first ``burnin`` samples. If omitted, the maximum of
            ``primes`` is used.
        primes (array_like, optional):
            The (non-)prime base to calculate values along each axis. If
            omitted, growing prime values starting from 2 will be used.

    Returns (numpy.ndarray):
        Hammersley set with ``shape == (dim, order)``.
    """
    if dim == 1:
        return create_halton_samples(
            order=order, dim=1, burnin=burnin, primes=primes)
    out = numpy.empty((dim, order), dtype=float)
    out[:dim-1] = create_halton_samples(
        order=order, dim=dim-1, burnin=burnin, primes=primes)
    out[dim-1] = numpy.linspace(0, 1, order+2)[1:-1]
    return out
"""
Generate samples from `low-discrepancy sequences`_.

In mathematics, a `low-discrepancy sequence`_ is a sequence with the property
that for all values of N, its subsequence x1, ..., xN has a low discrepancy.

Roughly speaking, the discrepancy of a sequence is low if the proportion of
points in the sequence falling into an arbitrary set B is close to proportional
to the measure of B, as would happen on average (but not for particular
samples) in the case of an equidistributed sequence. Specific definitions of
discrepancy differ regarding the choice of B (hyperspheres, hypercubes, etc.)
and how the discrepancy for every B is computed (usually normalized) and
combined (usually by taking the worst value).

Low-discrepancy sequences are also called quasi-random or sub-random sequences,
due to their common use as a replacement of uniformly distributed random
numbers. The "quasi" modifier is used to denote more clearly that the values of
a low-discrepancy sequence are neither random nor pseudorandom, but such
sequences share some properties of random variables and in certain applications
such as the quasi-Monte Carlo method their lower discrepancy is an important
advantage.

.. low-discrepancy sequence: https://en.wikipedia.org/wiki/Low-discrepancy_sequence
"""
"""
Create samples from a Korobov lattice.

Examples usage
--------------

Normal usage::

    >>> print(create_korobov_samples(order=4, dim=2))
    [[ 0.2  0.4  0.6  0.8]
     [ 0.4  0.8  0.2  0.6]]

With custom number base::

    >>> print(create_korobov_samples(order=4, dim=2, base=3))
    [[ 0.2  0.4  0.6  0.8]
     [ 0.6  0.2  0.8  0.4]]
"""
import numpy


def create_korobov_samples(order, dim, base=17797):
    """
    Create Korobov lattice samples.

    Args:
        order (int):
            The order of the Korobov latice. Defines the number of
            samples.
        dim (int):
            The number of dimensions in the output.
        base (int):
            The number based used to calculate the distribution of values.

    Returns (numpy.ndarray):
        Korobov lattice with ``shape == (dim, order)``
    """
    values = numpy.empty(dim)
    values[0] = 1
    for idx in range(1, dim):
        values[idx] = base*values[idx-1] % (order+1)

    grid = numpy.mgrid[:dim, :order+1]
    out = values[grid[0]] * (grid[1]+1) / (order+1.) % 1.
    return out[:, :order]
"""
Create all primes bellow a certain threshold.

Examples::

    >>> print(create_primes(1))
    []
    >>> print(create_primes(2))
    [2]
    >>> print(create_primes(3))
    [2, 3]
    >>> print(create_primes(20))
    [2, 3, 5, 7, 11, 13, 17, 19]
"""


def create_primes(threshold):
    """
    Generate prime values using sieve of Eratosthenes method.

    Args:
        threshold (int):
            The upper bound for the size of the prime values.

    Returns (List[int]):
        All primes from 2 and up to ``threshold``.
    """
    if threshold == 2:
        return [2]

    elif threshold < 2:
        return []

    numbers = list(range(3, threshold+1, 2))
    root_of_threshold = threshold ** 0.5
    half = int((threshold+1)/2-1)
    idx = 0
    counter = 3
    while counter <= root_of_threshold:
        if numbers[idx]:
            idy = int((counter*counter-3)/2)
            numbers[idy] = 0
            while idy < half:
                numbers[idy] = 0
                idy += counter
        idx += 1
        counter = 2*idx+3
    return [2] + [number for number in numbers if number]
"""
Generates samples from the `Sobol sequence`_.

Sobol sequences (also called LP_T sequences or (t, s) sequences in base 2) are
an example of quasi-random low-discrepancy sequences. They were first
introduced by the Russian mathematician Ilya M. Sobol in 1967.

These sequences use a base of two to form successively finer uniform partitions
of the unit interval and then reorder the coordinates in each dimension.

Example usage
-------------

Standard usage::

    >>> set_state(1000)
    >>> print(create_sobol_samples(order=2, dim=2))
    [[ 0.21972656  0.71972656  0.96972656]
     [ 0.09667969  0.59667969  0.34667969]]
    >>> print(create_sobol_samples(order=2, dim=2))
    [[ 0.46972656  0.34472656  0.84472656]
     [ 0.84667969  0.47167969  0.97167969]]
    >>> print(create_sobol_samples(order=2, dim=3))
    [[ 0.59472656  0.09472656  0.06347656]
     [ 0.22167969  0.72167969  0.19042969]
     [ 0.92285156  0.42285156  0.01660156]]
    >>> print(create_sobol_samples(order=2, dim=6))
    [[ 0.56347656  0.81347656  0.31347656]
     [ 0.69042969  0.44042969  0.94042969]
     [ 0.51660156  0.76660156  0.26660156]
     [ 0.17675781  0.92675781  0.42675781]
     [ 0.05371094  0.30371094  0.80371094]
     [ 0.25292969  0.50292969  0.00292969]]

Licence
-------

This code is distributed under the GNU LGPL license.

The routine adapts the ideas of Antonov and Saleev. Original FORTRAN77 version
by Bennett Fox. MATLAB version by John Burkardt. PYTHON version by Corrado
Chisari.

Papers::

    Antonov, Saleev,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 19, 1980, pages 252 - 256.

    Paul Bratley, Bennett Fox,
    Algorithm 659:
    Implementing Sobol's Quasirandom Sequence Generator,
    ACM Transactions on Mathematical Software,
    Volume 14, Number 1, pages 88-100, 1988.

    Bennett Fox,
    Algorithm 647:
    Implementation and Relative Efficiency of Quasirandom
    Sequence Generators,
    ACM Transactions on Mathematical Software,
    Volume 12, Number 4, pages 362-376, 1986.

    Ilya Sobol,
    USSR Computational Mathematics and Mathematical Physics,
    Volume 16, pages 236-242, 1977.

    Ilya Sobol, Levitan,
    The Production of Points Uniformly Distributed in a Multidimensional
    Cube (in Russian),
    Preprint IPM Akad. Nauk SSSR,
    Number 40, Moscow 1976.

.. Sobel sequence: https://en.wikipedia.org/wiki/Sobol_sequence
"""
import math

import numpy


RANDOM_SEED = 1
DIM_MAX = 40
LOG_MAX = 30

SOURCE_SAMPLES = numpy.zeros((DIM_MAX, LOG_MAX), dtype=int)
SOURCE_SAMPLES[0:40, 0] = 1
SOURCE_SAMPLES[2:40, 1] = (
    1, 3, 1, 3, 1, 3, 3, 1, 3, 1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3, 3, 1, 3,
    1, 3, 1, 3, 1, 1, 3, 1, 3, 1, 3, 1, 3
)
SOURCE_SAMPLES[3:40, 2] = (
    7, 5, 1, 3, 3, 7, 5, 5, 7, 7, 1, 3, 3, 7, 5, 1, 1, 5, 3, 3, 1, 7, 5, 1, 3,
    3, 7, 5, 1, 1, 5, 7, 7, 5, 1, 3, 3
)
SOURCE_SAMPLES[5:40, 3] = (
    1, 7, 9, 13, 11, 1, 3, 7, 9, 5, 13, 13, 11, 3, 15, 5, 3, 15, 7, 9, 13, 9,
    1, 11, 7, 5, 15, 1, 15, 11, 5, 3, 1, 7, 9
)
SOURCE_SAMPLES[7:40, 4] = (
    9, 3, 27, 15, 29, 21, 23, 19, 11, 25, 7, 13, 17, 1, 25, 29, 3, 31, 11, 5,
    23, 27, 19, 21, 5, 1, 17, 13, 7, 15, 9, 31, 9
)
SOURCE_SAMPLES[13:40, 5] = (
    37, 33, 7, 5, 11, 39, 63, 27, 17, 15, 23, 29, 3, 21, 13, 31, 25, 9, 49, 33,
    19, 29, 11, 19, 27, 15, 25
)
SOURCE_SAMPLES[19:40, 6] = (
    13, 33, 115, 41, 79, 17, 29, 119, 75, 73, 105, 7, 59, 65, 21, 3, 113, 61,
    89, 45, 107
)
SOURCE_SAMPLES[37:40, 7] = (7, 23, 39)
POLY = (
    1, 3, 7, 11, 13, 19, 25, 37, 59, 47, 61, 55, 41, 67, 97, 91, 109, 103,
    115, 131, 193, 137, 145, 143, 241, 157, 185, 167, 229, 171, 213, 191,
    253, 203, 211, 239, 247, 285, 369, 299
)


def set_state(seed_value=None, step=None):
    """Set random seed."""
    global RANDOM_SEED  # pylint: disable=global-statement
    if seed_value is not None:
        RANDOM_SEED = seed_value
    if step is not None:
        RANDOM_SEED += step


def create_sobol_samples(order, dim, seed=None):
    """
    Args:
        order (int):
            Number of unique samples to generate
        dim (int):
            Number of spacial dimensions. Must satisfy ``0 < dim < 41``.
        seed (int, optional):
            Starting seed. Non-positive values are treated as 1. If omitted,
            consequtive samples are used.

    Returns:
        quasi (numpy.ndarray):
            Quasi-random vector with ``shape == (dim, order+1)``.
    """
    assert 0 < dim < DIM_MAX, "dim in [1, 40]"

    # global RANDOM_SEED  # pylint: disable=global-statement
    # if seed is None:
    #     seed = RANDOM_SEED
    # RANDOM_SEED += order+1
    set_state(seed_value=seed)
    seed = RANDOM_SEED
    set_state(step=order+1)

    # Initialize row 1 of V.
    samples = SOURCE_SAMPLES.copy()
    maxcol = int(math.log(2**LOG_MAX-1, 2))+1
    samples[0, 0:maxcol] = 1

    # Initialize the remaining rows of V.
    for idx in range(1, dim):

        # The bits of the integer POLY(I) gives the form of polynomial:
        degree = int(math.log(POLY[idx], 2))

        #Expand this bit pattern to separate components:
        includ = numpy.array([val == "1" for val in bin(POLY[idx])[-degree:]])

        #Calculate the remaining elements of row I as explained
        #in Bratley and Fox, section 2.
        for idy in range(degree+1, maxcol+1):
            newv = samples[idx, idy-degree-1].item()
            base = 1
            for idz in range(1, degree+1):
                base *= 2
                if includ[idz-1]:
                    newv = newv ^ base * samples[idx, idy-idz-1].item()
            samples[idx, idy-1] = newv

    samples = samples[:dim]

    # Multiply columns of V by appropriate power of 2.
    samples *= 2**(numpy.arange(maxcol, 0, -1, dtype=int))

    #RECIPD is 1/(common denominator of the elements in V).
    recipd = 0.5**(maxcol+1)
    lastq = numpy.zeros(dim, dtype=int)

    seed = int(seed) if seed > 1 else 1

    for seed_ in range(seed):
        lowbit = len(bin(seed_)[2:].split("0")[-1])
        lastq[:] = lastq ^ samples[:, lowbit]

    #Calculate the new components of QUASI.
    quasi = numpy.empty((dim, order+1))
    for idx in range(order+1):
        lowbit = len(bin(seed+idx)[2:].split("0")[-1])
        quasi[:, idx] = lastq * recipd
        lastq[:] = lastq ^ samples[:, lowbit]

    return quasi
"""
Create `Van Der Corput` low discrepancy sequence samples.

A van der Corput sequence is an example of the simplest one-dimensional
low-discrepancy sequence over the unit interval; it was first described in 1935
by the Dutch mathematician J. G. van der Corput. It is constructed by reversing
the base-n representation of the sequence of natural numbers (1, 2, 3, ...).

In practice, use Halton sequence instead of Van Der Corput, as it is the
same, but generalized to work in multiple dimensions.

Example usage
-------------

Using base 10::

    >>> print(create_van_der_corput_samples(range(11), number_base=10))
    [ 0.1   0.2   0.3   0.4   0.5   0.6   0.7   0.8   0.9   0.01  0.11]

Using base 2::

    >>> print(create_van_der_corput_samples(range(8), number_base=2))
    [ 0.5     0.25    0.75    0.125   0.625   0.375   0.875   0.0625]

.. Van Der Corput: https://en.wikipedia.org/wiki/Van_der_Corput_sequence
"""
import numpy


def create_van_der_corput_samples(idx, number_base=2):
    """
    Van der Corput samples.

    Args:
        idx (int, array_like):
            The index of the sequence. If array is provided, all values in
            array is returned.
        number_base (int):
            The numerical base from where to create the samples from.

    Returns (float, numpy.ndarray):
        Van der Corput samples.
    """
    assert number_base > 1

    idx = numpy.asarray(idx).flatten() + 1
    out = numpy.zeros(len(idx), dtype=float)

    base = float(number_base)
    active = numpy.ones(len(idx), dtype=bool)
    while numpy.any(active):
        out[active] += (idx[active] % number_base)/base
        idx = idx / number_base
        idx = idx.astype(int)
        base *= number_base
        active = idx > 0
    return out
