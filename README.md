# Functionax

Functionax provides routines to implement discretized infinite-dimensional
optimization algorithms. Prime application is neural network training involving
PDE terms.

Caution: The package is far from a first workable version.

## Examples
- Newton in Function space, proposed in [arxiv](https://arxiv.org/abs/2302.13163).
- Lagrange-Newton for saddle point problems, proposed as CPINNs in 
    [arxiv](https://arxiv.org/abs/2204.11144).


## Developer guide

To contribute follow the description below.

## Setup Conda Environment

You can set up the `conda` environment and activate it

```bash
conda env create --file .conda_env.yaml
conda activate sample_project
```

## No Conda: Editable Install with PIP

In case you are not using conda you can install the package 
in editable mode using:

```bash
pip install -e ."[lint,test]"
```

## Linting and Formatting

We use ruff for linting and formatting. Automated checks are carried out through github
workflows. To run linting and formating locally use the following.

```bash
ruff check .
ruff format .
```

## Github Workflows

Upon pushing checks are carried out through Github Workflows.
The checks run the ruff linter and execute the tests.