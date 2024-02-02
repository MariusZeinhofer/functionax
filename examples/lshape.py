from functionax.domains import LShape, LShapeBoundary
from jax import random, vmap, jit, grad, jacrev, jvp, vjp

import jax.numpy as jnp
from utils import mlp, init_params, laplace
from matplotlib import pyplot as plt
from jax.flatten_util import ravel_pytree


# set the domain and the points for training and the points for evaluating
domain = LShape()
x_Omega = domain.sample_uniform(random.PRNGKey(0), N=1000)
x_Eval = domain.sample_uniform(random.PRNGKey(1), N=30000)
# set the domian boundary and its training points
domain_boundary = LShapeBoundary()
x_Gamma = domain_boundary.sample_uniform(random.PRNGKey(2), side_number=None, N=50)

# define the neural network functions
activation = lambda x: jnp.tanh(x)
layer_sizes = [2, 20, 20, 20, 1]
params = init_params(layer_sizes, random.PRNGKey(3))
model = mlp(activation)
# call jax ravel_pytree to flatten the params to one array
# unravel is a callable for unflattening back to the same dimension
f_params, unravel = ravel_pytree(params)

## import the 3exact solution denoted  by u_star and the right hand side denoted by f
from lshape_utils import *

u_star_v = vmap(u_star, (0))

# interior residual and loss function
interior_res = lambda params, x: laplace(model, argnum=1)(params, x) + f(x)
v_interior_res = vmap(interior_res, (None, 0))


def interior_loss(params, x_Omega):
    return 1.0 / 2.0 * jnp.mean(v_interior_res(params, x_Omega) ** 2)


# boundary residual and loss function
boundary_res = lambda params, x: model(params, x) - u_star(x)
v_boundary_res = vmap(boundary_res, (None, 0))


def boundary_loss(params, x_Gamma):
    return 1.0 / 2.0 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)


# total loss function
@jit
def loss(params, x_Omega):
    return interior_loss(params, x_Omega) + boundary_loss(params, x_Gamma)
