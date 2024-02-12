"""
Contains several needed utility functions:

1. implementation of a MLP, i.e., a fully connected model.

"""
import jax.numpy as jnp
from jax import random, hessian, jit, vmap, jacrev, jacfwd
from jax.flatten_util import ravel_pytree



# ------Copied from Jax documentation-----
def random_layer_params(m: int, n: int, key, scale: float = 1e-1):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


# Initialize all layers for a fully-connected neural network with sizes "sizes"
def init_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [
        random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)
    ]


# ----------Copy ends----------------------


def mlp(activation):
    """
    Parameters
    ----------
    activation function

    Returns
    -------
    a function which takes as input the parameters and the physical value x
    """

    def model(params, inpt):
        hidden = inpt
        for w, b in params[:-1]:
            outputs = jnp.dot(w, hidden) + b
            hidden = activation(outputs)

        final_w, final_b = params[-1]
        return jnp.dot(final_w, hidden) + final_b

    return model


def laplace(func, argnum=0):
    """
    Computes laplacian of func with respect to the argument argnum.

    Parameters
    ----------
    func: Callable
        Function whose laplacian should be computed.

    argnum: int
        Argument number wrt which laplacian should be computed.

    Returns
    -------
    Callable of same signature as func.

    Issues
    ------
    Vector valued func. So far not tested if this function works
    appropriately for vector valued functions. We need an
    implementation that does this.

    """
    hesse = hessian(func, argnum)
    return lambda *args, **kwargs: jnp.trace(
        hesse(*args, **kwargs),
        axis1=-2,
        axis2=-1,
    )


def grid_line_search_factory(loss, x_Omega, steps):
    """
    Parameters
    ----------
    func:    callable loss function

    x_Omega: points inside the domain

    steps:   an array of step sizes

    Returns
    -------
    a callable

    """

    def loss_at_step(step, params, tangent_params):
        updated_params = [
            (w - step * dw, b - step * db)
            for (w, b), (dw, db) in zip(params, tangent_params)
        ]
        return loss(updated_params, x_Omega)

    v_loss_at_steps = jit(vmap(loss_at_step, (0, None, None)))

    @jit
    def grid_line_search_update(params, tangent_params):
        losses = v_loss_at_steps(steps, params, tangent_params)
        step_size = steps[jnp.argmin(losses)]
        return [
            (w - step_size * dw, b - step_size * db)
            for (w, b), (dw, db) in zip(params, tangent_params)
        ], step_size

    return grid_line_search_update



def gram_factory(v_residual):
    """
    ...
    
    Parameters
    ----------
    residual: Callable
        Of signature (PyTree, (d_in,)) -> (d_res,) where PyTree may or
        may not be flattened.

    Returns 
    -----
    A callable gramian function with input: params and points 
    """

    def gramian(*params, x):
        """
        ...

        Parameters
        ----------
        params:
            one or more PyTrees.

        x:
            Array of shape (N, d_in)

        """
        # flatten params to enable correct jacobian compuations
        f_params, unravel = ravel_pytree(params[0])

        # For long flat jacobians we use reverse mode autodiff.
        if len(f_params) > len(x):
            jac = jacrev

        # for tall slim jacobians we use forward mode autodiff.
        else:
            jac = jacfwd

        # Compute the jacobian on batched data
        J = jac(lambda f_params, x: v_residual(unravel(f_params), x), argnums=0)(f_params, x=x)

        return 1.0 / len(x) * jnp.transpose(J) @ J

    return gramian



def accumulate(splits, argname):
    def _accumulate(f):
        def T_f(*args, **kwargs):
            Xs = jnp.array_split(kwargs[argname], splits, axis=0)
            N_total = len(kwargs[argname])

            sums = 0
            
            for x in Xs:
                N_batch = len(x)
                kwargs[argname] = x
                sums += (N_batch/N_total) * f(*args, **kwargs)
            return sums
        return T_f
    return _accumulate 