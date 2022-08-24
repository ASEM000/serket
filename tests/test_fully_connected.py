import pytest

from serket.nn import PFNN, Linear


def test_PFNN():

    with pytest.raises(ValueError):
        PFNN([1, 2, [3, 2], 3, 2])

    with pytest.raises(ValueError):
        PFNN([1, 2, [3, 2], 3, 2])

    with pytest.raises(AssertionError):
        PFNN([1, 2, [3, 2], 3, 1])

    with pytest.raises(TypeError):
        PFNN([1, 2, [3, "a"], 3, 2])

    assert PFNN([1, 2]).layers == [[Linear(1, 1)]] * 2
    assert PFNN([1, [2, 3], 2]).layers == [
        [Linear(1, 2), Linear(2, 1)],
        [Linear(1, 3), Linear(3, 1)],
    ]
    assert PFNN([1, 2, 3]).layers == [[Linear(1, 2), Linear(2, 1)]] * 3
