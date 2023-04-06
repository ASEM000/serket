# import jax.numpy as jnp
# import pytest

# from serket.nn import PFNN, Linear


# def test_PFNN():

#     with pytest.raises(ValueError):
#         PFNN([1, 2, [3, 2], 3, 2])

#     with pytest.raises(ValueError):
#         PFNN([1, 2, [3, 2], 3, 2])

#     with pytest.raises(AssertionError):
#         PFNN([1, 2, [3, 2], 3, 1])

#     with pytest.raises(TypeError):
#         PFNN([1, 2, [3, "a"], 3, 2])

#     assert PFNN([1, 2]).layers == [[Linear(1, 1)]] * 2
#     assert PFNN([1, [2, 3], 2]).layers == [
#         [Linear(1, 2), Linear(2, 1)],
#         [Linear(1, 3), Linear(3, 1)],
#     ]
#     assert PFNN([1, 2, 3]).layers == [[Linear(1, 2), Linear(2, 1)]] * 3

#     assert PFNN([1, [2, 2, 2], 3]).layers == [[Linear(1, 2), Linear(2, 1)]] * 3
#     assert PFNN([1, [2, 3, 4], 3]).layers == [
#         [Linear(1, 2), Linear(2, 1)],
#         [Linear(1, 3), Linear(3, 1)],
#         [Linear(1, 4), Linear(4, 1)],
#     ]

#     assert PFNN([1, [2, 3, 4],[1,1,1], 3]).layers == [
#         [Linear(1, 2), Linear(2, 1),Linear(1, 1)],
#         [Linear(1, 3), Linear(3, 1),Linear(1, 1)],
#         [Linear(1, 4), Linear(4, 1),Linear(1, 1)],
#     ]

#     with pytest.raises(TypeError):
#         PFNN([1, [2, "a", 2], 3])

#     assert PFNN([1, [2, 3], 2])(jnp.ones([1, 1])).shape == (1, 2)

#     with pytest.raises(AssertionError):
#         PFNN([1,[2,3],[3,4,5],2])

#     with pytest.raises(TypeError):
#         PFNN([1,"a",[3,4,5],2])
