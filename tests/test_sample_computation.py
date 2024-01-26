"""An example test."""

from functionax.sample_computation import quadratics


def test_answer():
    """We test the quadratics function at one input."""
    assert quadratics(3) == 9
