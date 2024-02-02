""" 
Contains functions needed to define 
the singular solution for the Lshape domain 

"""

from functionax.domains import LShape, LShapeBoundary
from jax import random, vmap, jit, grad, jacrev, jvp, vjp

import jax.numpy as jnp
from utils import mlp, init_params, laplace
from matplotlib import pyplot as plt
from jax.flatten_util import ravel_pytree


# from cartesian to polar
def cart2pol(xy):
    x = -xy[0]
    y = -xy[1]

    r = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return jnp.array([r, phi + jnp.pi])


v_cart2pol = vmap(cart2pol, (0))


# from polar to cartesian
def pol2cart(rphi):
    r = rphi[0]
    phi = rphi[1]

    x = r * jnp.cos(phi)
    y = r * jnp.sin(phi)
    return jnp.array([x, y])


v_pol2cart = vmap(pol2cart, (0))


# u = w + eta * s


def s(xy):
    """the singularity"""
    rtheta = cart2pol(xy)
    r = rtheta[0]
    theta = rtheta[1]
    return ((r**2.0) ** (1.0 / 3.0)) * jnp.sin(2.0 * theta / 3.0)


v_s = vmap(s, (0))


R = 1.0 / 2.0


def eta(x, R):
    r = cart2pol(x)[0]
    return jnp.select(
        [(0.0 <= r) & (r < R / 2.0), (R / 2.0 <= r) & (r <= R), R < r],
        [
            1.0,
            15.0
            / 16.0
            * (
                8.0 / 15.0
                - (4 * r / R - 3)
                + 2 / 3 * (4 * r / R - 3) ** 3
                - 1 / 5 * (4 * r / R - 3) ** 5
            ),
            0.0,
        ],
        default=jnp.nan,
    )


v_eta = vmap(eta, (0, None))


def f(xy):
    x = xy[0]
    y = xy[1]

    return jnp.select(
        [(-1.0 <= y) & (y < 0.0), (0.0 < y) & (y <= 1.0)],
        [
            jnp.sin(2.0 * jnp.pi * x)
            * (2.0 * jnp.pi**2 * (y**2 + 2 * y) * (y**2 - 1) - (6 * y**2 + 6 * y - 1))
            - laplace(lambda xy, R: eta(xy, R) * s(xy))(xy, R),
            jnp.sin(2.0 * jnp.pi * x)
            * (
                2.0 * jnp.pi**2 * (-(y**2) + 2 * y) * (y**2 - 1)
                - (-6 * y**2 + 6 * y + 1)
            )
            - laplace(lambda xy, R: eta(xy, R) * s(xy))(xy, R),
        ],
        default=0.0,
    )


def w(xy):
    x = xy[0]
    y = xy[1]

    return jnp.select(
        [(-1.0 <= y) & (y < 0.0), (0.0 <= y) & (y <= 1.0)],
        [
            jnp.sin(2 * jnp.pi * x) * (0.5 * y**2 + y) * (y**2 - 1),
            jnp.sin(2 * jnp.pi * x) * (-0.5 * y**2 + y) * (y**2 - 1),
        ],
        default=0.0,
    )


def u_star(xy):
    return w(xy) + eta(xy, R) * s(xy)


u_star_v = vmap(u_star, (0))


if __name__ == "__main__":
    ## test the solution
    domain = LShape()
    x_Eval = domain.sample_uniform(random.PRNGKey(1), N=30000)
    plt.scatter(x_Eval[:, 0], x_Eval[:, 1], s=10, c=jnp.array(u_star_v(x_Eval)))
    plt.gca().set_aspect(1.0)
    plt.colorbar()

    plt.show()
