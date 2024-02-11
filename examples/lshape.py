from functionax.domains import LShape, LShapeBoundary
from jax import random, vmap, jit, grad, jacrev, jvp, vjp
from jax.numpy.linalg import lstsq

import jax.numpy as jnp
from utils import mlp, init_params, laplace, grid_line_search_factory, gram_factory, accumulate
from matplotlib import pyplot as plt
from jax.flatten_util import ravel_pytree
import jax 

jax.config.update("jax_enable_x64", True)

# set the domain and the points for training and the points for evaluating
domain = LShape()
x_Omega = domain.sample_uniform(random.PRNGKey(0), N=5000)
x_Eval = domain.sample_uniform(random.PRNGKey(1), N=5000)
# set the domian boundary and its training points
domain_boundary = LShapeBoundary()
x_Gamma = domain_boundary.sample_uniform(random.PRNGKey(2), side_number=None, N=40)

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
interior_res = lambda params, x: jnp.reshape(laplace(model, argnum=1)(params, x) + f(x), ())
v_interior_res = vmap(interior_res, (None, 0))

def interior_loss(params, x_Omega):
    return 1.0 / 2.0 * jnp.mean(v_interior_res(params, x_Omega) ** 2)


# boundary residual and loss function
boundary_res = lambda params, x: jnp.reshape(model(params, x) - u_star(x), ())
v_boundary_res = vmap(boundary_res, (None, 0))


def boundary_loss(params, x_Gamma):
    return 1.0 / 2.0 * jnp.mean(v_boundary_res(params, x_Gamma) ** 2)


# total loss function
@jit
def loss(params, x_Omega):
    return interior_loss(params, x_Omega) + boundary_loss(params, x_Gamma)


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
grid_line_search_update = grid_line_search_factory(loss, x_Omega, steps)


# errors
error = lambda x: jnp.reshape(model(params, x) - u_star(x), ())
v_error = vmap(error, (0))
v_error_abs_grad = vmap(lambda x: jnp.sum(jacrev(error)(x) ** 2.0) ** 0.5)




def l2_norm(f, x_eval):
    return (1 / 3) * jnp.mean((f(x_eval)) ** 2.0) ** 0.5


l2_error  = l2_norm(v_error, x_Eval)
h1_error  = l2_error + l2_norm(v_error_abs_grad, x_Eval)

gram_int  = jit(accumulate(30,'x')(gram_factory(v_interior_res)))
gram_bdry = jit(gram_factory(v_boundary_res))

VERBOSE = True
LM = 1e-04
# natural gradient descent with line search
for iteration in range(500):
    
    # compute gradient of loss
    grads = grad(loss)(params,x_Omega)
    f_grads = ravel_pytree(grads)[0]

    # assemble gramian
    G_int  = gram_int(params,  x= x_Omega)
    G_bdry = gram_bdry(params, x= x_Gamma)
    G      = G_int + G_bdry

    # Marquardt-Levenberg
    Id = jnp.identity(len(G))
    G = jnp.min(jnp.array([loss(params,x_Omega), LM])) * Id + G

    # compute natural gradient
    f_nat_grad = lstsq(G, f_grads, rcond=-1)[0]
    nat_grad = unravel(f_nat_grad)
    
    # one step of NGD
    params, actual_step = grid_line_search_update(params, nat_grad)

    if iteration % 50 == 0:
        # errors
        l2_error = l2_norm(v_error, x_Eval)
        h1_error = l2_error + l2_norm(v_error_abs_grad, x_Eval)

        if VERBOSE == True:
            print(
                f'NG Iteration: {iteration} with loss: {loss(params,x_Omega)} with error '
                f'L2: {l2_error} and error H1: {h1_error} and step: {actual_step}'
            )

l2_error = l2_norm(v_error, x_Eval)
h1_error = l2_error + l2_norm(v_error_abs_grad, x_Eval)
print(f'POISSON EQUATION: loss {loss(params,x_Omega)}, L2 {l2_error}, H1 {h1_error}')
