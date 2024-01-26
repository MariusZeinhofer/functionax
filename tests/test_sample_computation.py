"""An example test."""

from sample.sample_computation import quadratics


def test_answer():
    """We test the quadratics function at one input."""
    assert quadratics(3) == 90
