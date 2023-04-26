import jax.numpy as jnp
import pytest

from serket.nn import PFNN  # , Linear


def test_PFNN():
    with pytest.raises(ValueError, match="Cannot join paths"):
        PFNN([1, 2, [3, 2], 3, 2])

    with pytest.raises(ValueError):
        PFNN([1, 2, [3, 2], 3, 2])

    with pytest.raises(ValueError):
        PFNN([1, 2, [3, 2], 3, 1])

    with pytest.raises(TypeError):
        PFNN([1, 2, [3, "a"], 3, 2])

    with pytest.raises(ValueError):
        PFNN([1, 2])

    with pytest.raises(TypeError):
        PFNN([1, [2, "a", 2], 3])

    assert PFNN([1, [2, 3], 2])(jnp.ones([1, 1])).shape == (1, 2)

    with pytest.raises(ValueError):
        PFNN([1, [2, 3], [3, 4, 5], 2])

    with pytest.raises(TypeError):
        PFNN([1, "a", [3, 4, 5], 2])
