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
x_Omega = domain.sample_uniform(random.PRNGKey(0), N=10000)
x_Eval = domain.sample_uniform(random.PRNGKey(1), N=20000)
# set the domian boundary and its training points
domain_boundary = LShapeBoundary()
N_Gamma = 40
x_Gamma = domain_boundary.sample_uniform(random.PRNGKey(2), side_number=None, N=N_Gamma)

# define the neural network functions
activation = lambda x: jnp.tanh(x)
layer_sizes = [2, 20,20,20, 1]
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


from jax import lax

## H1/2 contribution
import numpy as np

x_h12 = [] 

for side_number in range(6): 
  x1 = domain_boundary.sample_uniform(random.PRNGKey(2), side_number = side_number, N=N_Gamma) 
  x2 = domain_boundary.sample_uniform(random.PRNGKey(3), side_number = side_number, N=N_Gamma)
  x12 = np.stack([[[x,y] for x,y in list(zip(x1,x2))]])
  x12 = jnp.reshape(x12, (N_Gamma,2,2))
  x_h12.append(x12) 



def h12_func_cond(params, x): 
  return lax.cond(jnp.sum(jnp.abs(x[0]-x[1])) > 1e-8, lambda params,x : jnp.reshape((model(params,x[0]) -u_star(x[0]) - model(params,x[1]) + u_star(x[1]))/(jnp.sum(jnp.abs(x[0]-x[1]))),()),
                  lambda params, x: 0.0, 
                  params, x
                  )
v_h12_cond = vmap(h12_func_cond, (None, 0))



def h12_loss(params): 
  loss = 0.0 
  for side_number in range(6): 
    loss += 1./2.*jnp.mean(v_h12_cond(params, x_h12[side_number])**2) 
  return loss 


# total loss function
@jit
def loss(params, x_Omega):
    return interior_loss(params, x_Omega) + boundary_loss(params, x_Gamma) + h12_loss(params)


# set up grid line search
grid = jnp.linspace(0, 30, 31)
steps = 0.5**grid
grid_line_search_update = grid_line_search_factory(loss, x_Omega, steps)


# errors
error = lambda x: jnp.reshape(model(params, x) - u_star(x), ())
v_error = vmap(error, (0))
v_error_abs_grad = vmap(lambda x: jnp.sum(jacrev(error)(x) ** 2.0) ** 0.5)


def l2_norm(f, x_eval):
    return (1 / 3) * (jnp.mean((f(x_eval)) ** 2.0) ** 0.5)


l2_error  = l2_norm(v_error, x_Eval)
l2_norm_true = l2_norm(u_star_v, x_Eval) 
h1_error  = l2_error + l2_norm(v_error_abs_grad, x_Eval)

gram_int  = jit(accumulate(30,'x')(gram_factory(v_interior_res)))
gram_bdry = jit(gram_factory(v_boundary_res))
gram_h12_bdry = jit(gram_factory(v_h12_cond))

VERBOSE = True
LM = 1e-07
adaptive_interior = False 
key = random.PRNGKey(10) 
key, subkey = random.split(key)
# natural gradient descent with line search
for iteration in range(1800):
    
    # compute gradient of loss
    grads = grad(loss)(params,x_Omega)
    f_grads = ravel_pytree(grads)[0]

    # assemble gramian
    G_int  = gram_int(params,  x= x_Omega)
    G_bdry = gram_bdry(params, x= x_Gamma)
    G      = G_int + G_bdry
    for i in range(6): 
      G +=  gram_h12_bdry(params, x = x_h12[i])
    # Marquardt-Levenberg
    Id = jnp.identity(len(G))
    G = jnp.min(jnp.array([loss(params,x_Omega), LM])) * Id + G
    grid_line_search_update = grid_line_search_factory(loss, x_Omega, steps)

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
                f'L2: {l2_error} , L2 weighted: {l2_error/l2_norm_true}, and error H1: {h1_error} and step: {actual_step}'
            )
    
    if (iteration % 100 == 0) & (iteration > 0) & (adaptive_interior):
        fitness = jnp.abs(v_interior_res(params, x_Omega))
        # set the threshold
        threshold = jnp.mean(fitness)
        # remove non-fit collocation points
        mask = jnp.squeeze(jnp.where(fitness > threshold, False, True))
        x_fit = jnp.delete(x_Omega, mask, axis=0)

        # add new uniformly drawn collocation points to fill up
        N_fit = len(x_Omega) - len(x_fit)
        # draw points near the corner singularity
        if N_fit >= 2:
            x_add =  domain.sample_uniform(subkey, N=N_fit)
            x_Omega = jnp.concatenate([x_fit, x_add], axis=0)
            print(f"iteration: {iteration}", f"updated: {len(x_add)}", "points")
            jnp.save('{}{}{}'.format("adaptive_", (iteration/100) ,".npy") , x_Omega)
            # advance random number generator
            key, subkey= random.split(key)
    
l2_error = l2_norm(v_error, x_Eval)
h1_error = l2_error + l2_norm(v_error_abs_grad, x_Eval)
print(f'POISSON EQUATION: loss {loss(params,x_Omega)}, L2 {l2_error}, H1 {h1_error}')

visualize = True 
if visualize: 

    plt.scatter(x_Eval[:, 0], x_Eval[:, 1], s=50, c=vmap(model, (None, 0))(params, x_Eval))
    plt.gca().set_aspect(1.0)
    plt.colorbar()
    plt.show()

    plt.scatter(x_Eval[:, 0], x_Eval[:, 1], s=50, c=v_error(x_Eval))
    plt.gca().set_aspect(1.0)
    plt.colorbar()
    plt.show()
