import jax
import jax.random as jr
import jax.tree_util as jtu
import pytest

from serket.nn.callbacks import init_func_cb
from serket.nn.utils import canonicalize


def test_canonicalize_init_func():
    k = jr.PRNGKey(0)

    assert init_func_cb("he_normal")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("he_uniform")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("glorot_normal")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("glorot_uniform")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("lecun_normal")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("lecun_uniform")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("normal")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("uniform")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("ones")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("zeros")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("xavier_normal")(k, (2, 2)).shape == (2, 2)
    assert init_func_cb("xavier_uniform")(k, (2, 2)).shape == (2, 2)

    assert isinstance(init_func_cb(jax.nn.initializers.he_normal()), jtu.Partial)
    assert isinstance(init_func_cb(None), type(None))

    with pytest.raises(ValueError):
        init_func_cb("invalid")

    with pytest.raises(ValueError):
        init_func_cb(1)


def test_canonicalize():
    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)

    with pytest.raises(ValueError):
        canonicalize((3, 3), 3)

    with pytest.raises(ValueError):
        canonicalize((3, 3, 3), 2)

    with pytest.raises(ValueError):
        canonicalize((3, 3, 3), 1)

    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)

    assert canonicalize(3, 2) == (3, 3)
    assert canonicalize((3, 3), 2) == (3, 3)
    assert canonicalize((3, 3, 3), 3) == (3, 3, 3)


# def test_canonicalize_padding():
#     assert canonicalize(1, (3, 3)) == ((1, 1), (1, 1))
#     assert canonicalize(0, (3, 3)) == ((0, 0), (0, 0))
#     assert canonicalize(2, (3, 3)) == ((2, 2), (2, 2))

#     assert canonicalize((1, 1), (3, 3)) == ((1, 1), (1, 1))
#     assert canonicalize(((1, 1), (1, 1)), (3, 3)) == ((1, 1), (1, 1))
#     assert canonicalize(("same", "same"), (3, 3)) == ((1, 1), (1, 1))
#     assert canonicalize(("valid", "valid"), (3, 3)) == ((0, 0), (0, 0))
#     with pytest.raises(ValueError):
#         canonicalize(("invalid", "valid"), (3, 3))

#     with pytest.raises(ValueError):
#         canonicalize(("valid", "invalid"), (3, 3))

#     with pytest.raises(ValueError):
#         canonicalize(("invalid", ()), (3, 3))
