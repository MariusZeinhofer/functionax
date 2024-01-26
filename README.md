# Developer guide

How to use this sample project?

## Setup Conda Environment

You can set up the `conda` environment and activate it

```bash
conda env create --file .conda_env.yaml
conda activate sample_project
```

## Editable Install with PIP

In case you are not using conda you can install the package 
in editable mode using:

```bash
pip install -e ."[lint,test]"
```

## Linting and Formatting

We use ruff for linting and formatting:

```bash
ruff check .
ruff format .
```

## Github Workflows

Upon pushing checks are carried out through Github Workflows.
The checks run the ruff linter and execute the tests.