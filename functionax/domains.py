"""Includes method to sample from geometries."""

import abc

from typing import Any
from jaxtyping import Array, Float, jaxtyped
import jax.numpy as jnp
from jax.random import uniform, randint
from jax import random
import math
import jax 
jax.config.update("jax_enable_x64", True)

class Domain(metaclass=abc.ABCMeta):
    """Abstract base class for computational domains.

    This abstract baseclass enforces that subclasses implement the metods
    sample_uniform and measure.

    """

    @classmethod
    def __subclasshook__(cls, subclass):
        """Subclasses of Domain must implement the method subclass.sample_uniform."""
        return callable(subclass.sample_uniform)

    @abc.abstractmethod
    def sample_uniform(self, key, N):
        """Draws uniform points from the Domain.

        Args:
            key: an explicit random key. Use jax.random.PRNGKey
            N: number of points sampled.

        Returns:
            A tensor of shape (N, d), where d is the dimension of the domain.
        """


class Hyperrectangle(Domain):
    """A product of intervals in R^d.

    The hyperrectangle is specified as a product of intervals.
    For example

    intervals = ((0., 1.), (0., 1.), (0., 1.))

    is the unit cube in R^3. The assumption is that intervals
    is convertable to an array of shape (d, 2).

    Note that the method sample_grid is not provided in this class.
    The Hyperrectangle is potentially a high dimensional object.
    Deterministic integration points should be implemented in child
    classes.

    Args:
        intervals: An iterable of intervals, see example above.
    """

    def __init__(self, intervals):
        self._intervals = jnp.array(intervals)

        # is this code needed?
        l_bounds = None
        r_bounds = None

        if jnp.shape(self._intervals) == (2,):
            l_bounds = self._intervals[0]
            r_bounds = self._intervals[1]

        else:
            l_bounds = self._intervals[:, 0]
            r_bounds = self._intervals[:, 1]

        self._l_bounds = jnp.reshape(
            jnp.asarray(l_bounds, dtype=float),
            newshape=(-1),
        )

        self._r_bounds = jnp.reshape(
            jnp.asarray(r_bounds, dtype=float),
            newshape=(-1),
        )

        if len(self._l_bounds) != len(self._r_bounds):
            raise ValueError(
                f"[In constructor of Hyperrectangle]: intervals "
                f"is not convertable to an array of shape (d, 2)."
            )

        if not jnp.all(self._l_bounds < self._r_bounds):
            raise ValueError(
                f"[In constructor of Hyperrectangle]: The "
                f"lower bounds must be smaller than the upper bounds."
            )

        self._dimension = len(self._l_bounds)

    def measure(self) -> float:
        return jnp.product(self._r_bounds - self._l_bounds)

    @jaxtyped
    # @typechecker
    def sample_uniform(self, key: Any, N: int = 50): #-> Float[Array, "N d"]:
        """
        N uniformly drawn collocation points in the hyperrectangle.

        Args:
            key: A random key from jax.random.PRNGKey(<int>).
            N: Number of random points.
        Returns: An Array of shape (N, d) where N is the number of points and d is
            the spatial dimension.
        """
        return uniform(
            key,
            shape=(N, self._dimension),
            minval=jnp.broadcast_to(
                self._l_bounds,
                shape=(N, self._dimension),
            ),
            maxval=jnp.broadcast_to(
                self._r_bounds,
                shape=(N, self._dimension),
            ),
        )


class Rectangle(Hyperrectangle):
    """Contains methods to sample from a 2d Rectangle."""

    def __init__(self, intervals):
        """Initialize the rectangle via passing intervals.

        Args:
            intervals: Four numbers, format [(a, b), (c, d)] corresponds to the
            rectangle [a, b] x [c, d].
        """
        # this will save the member variables? double check.
        super().__init__(intervals)

    def sample_grid(self, grid_size):
        """Draw points from a grid.

        Args:
            grid_size: grid size parameter

        Returns:
            A tensor of shape (N, 2), where N depends on the grid size.
        """
        pass


class LShape(Domain):
    """The classical L-shaped domain. We leave out the 4th quadrant.

    The domain is ([-1, 1] x [-1, 1]) \ ([0, 1] x [-1, 0]).
    """

    def __init__(self):
        pass  # no init needed. Is this allowed?

    def sample_uniform(self, key, N):
        """Samples uniform from the L-shaped domain.

        See parent class for documentation.
        """
        n, r = N // 3, N % 3
        keys = random.split(key, 4)
        subdomain_index = randint(keys[0], shape=(), minval=0, maxval=3)
        points_per_subdomain = [n, n, n]
        points_per_subdomain[subdomain_index] += r

        N0 = points_per_subdomain[0]
        N1 = points_per_subdomain[1] + points_per_subdomain[2]

        X_0 = uniform(
            keys[1],
            shape=(N0, 2),
            minval=jnp.broadcast_to(
                jnp.array((0, 0)),
                shape=(N0, 2),
            ),
            maxval=jnp.broadcast_to(
                jnp.array((1, 1)),
                shape=(N0, 2),
            ),
        )
        X_1 = uniform(
            keys[2],
            shape=(N1, 2),
            minval=jnp.broadcast_to(
                jnp.array((-1, -1)),
                shape=(N1, 2),
            ),
            maxval=jnp.broadcast_to(
                jnp.array((0, 1)),
                shape=(N1, 2),
            ),
        )

        X = jnp.concatenate([X_0, X_1])
        return X


class LShapeBoundary(Domain):

    """The boundary of the classical L-shaped domain. We leave out the 4th quadrant.

    The domain is ([-1, 1] x [-1, 1]) \ ([0, 1] x [-1, 0]).

    The numbering is the following:

    ------- 4---------
    |                |
    |                3
    5        --- 2---
    |       |
    |       1
    ----0----

    Parameters
    ----------
    side_number: int or slice_object
        Default means the full boundary is returned. Indices or
        slices between 0 and 5 can be used to retrieve other boundaries.
    ----------
    """

    def __init__(self):
        pass

    def sample_uniform(
        self, key, side_number=slice(0, 8), N: int = 50
    ): # -> Float[Array, "N 2"]:
        keys = random.split(key, num=5)

        # side 0
        a_0 = -1
        b_0 = 0
        # number of points weighted by length of the interval
        M_0 = jnp.maximum(math.ceil((b_0 - a_0) * N), 1)
        points_0 = random.uniform(keys[0], (M_0, 1), minval=a_0, maxval=b_0)
        # the points on this side have y r= -1
        side0_y = -1 * jnp.ones(shape=(M_0, 1))
        side_0 = jnp.concatenate([points_0, side0_y], axis=1)

        # side 1 has the same number of points as side 0
        # the points on this side have x = 0
        points_1 = random.uniform(keys[1], (M_0, 1), minval=a_0, maxval=b_0)
        side1_x = 0.0 * jnp.ones(shape=(M_0, 1))
        side_1 = jnp.concatenate([side1_x, points_1], axis=1)

        # side 2
        a_1 = 0
        b_1 = 1
        # number of points weighted by length of the interval
        M_1 = jnp.maximum(math.ceil((b_1 - a_1) * N), 1)
        points_2 = random.uniform(keys[2], (M_1, 1), minval=a_1, maxval=b_1)
        # the points on this side have y = 0
        side2_y = 0.0 * jnp.ones(shape=(M_1, 1))
        side_2 = jnp.concatenate([points_2, side2_y], axis=1)

        # side 3
        # the points on this side have x = 1
        points_3 = random.uniform(keys[3], (M_1, 1), minval=a_1, maxval=b_1)
        side3_x = 1.0 * jnp.ones(shape=(M_1, 1))
        side_3 = jnp.concatenate([side3_x, points_3], axis=1)

        # side 4
        a_2 = -1
        b_2 = 1
        # number of points weighted by length of the interval
        M_2 = jnp.maximum(math.ceil((b_1 - a_1) * N), 1)
        points_4 = random.uniform(keys[4], (M_2, 1), minval=a_2, maxval=b_2)
        # the points on this side have y = 1
        side4_y = 1.0 * jnp.ones(shape=(M_2, 1))
        side_4 = jnp.concatenate([points_4, side4_y], axis=1)

        # side 5
        points_5 = random.uniform(keys[5], (M_2, 1), minval=a_2, maxval=b_2)
        # the points on this side have x = -  1
        side5_x = -1.0 * jnp.ones(shape=(M_2, 1))
        side_5 = jnp.concatenate([side5_x, points_5], axis=1)

        sides = [side_0, side_1, side_2, side_3, side_4, side_5]

        # of shape (n, 2)
        if side_number == None:
            return jnp.reshape(jnp.array(sides), (-1, 2))
        else:
            # print(jnp.array(sides[side_number]))
            return jnp.reshape(
                jnp.array(sides[side_number]),
                (-1, 2),
            )

    
