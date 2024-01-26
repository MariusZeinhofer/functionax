"""Includes method to sample from geometries."""


class Rectangle:
    """Contains methods to sample from a 2d Rectangle."""

    def __init__(self, intervals):
        """Initialize the rectangle via passing intervals.

        Args:
            intervals: Four numbers, format [(a, b), (c, d)] corresponds to the
            rectangle [a, b] x [c, d].
        """
        self._intervals = intervals

    def sample_uniform(self, key, N):
        """Draw uniform random points from the rectangle.

        Args:
            key: an explicit random key. Use jax.random.PRNGKey
            N: number of points sampled.

        Returns:
            A tensor of shape (N, 2).
        """
        pass

    def sample_grid(self, grid_size):
        """Draw points from a grid.

        Args:
            grid_size: grid size parameter

        Returns:
            A tensor of shape (N, 2), where N depends on the grid size.
        """
        pass
