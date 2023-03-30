from serket.nn import Lambda, Sequential


def test_sequential():
    model = Sequential((Lambda(lambda x: x),))
    assert model(1.0) == 1.0

    model = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)))
    assert model(1.0) == 3.0

    model = Sequential((lambda x, key: x,))
    assert model(1.0) == 1.0


def test_sequential_getitem():
    model = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)))
    assert model[0](1.0) == 2.0
    assert model[1](1.0) == 2.0
    assert model[0:1](1.0) == 2.0
    assert model[1:2](1.0) == 2.0
    assert model[0:2](1.0) == 3.0


def test_sequential_len():
    model = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)))
    assert len(model) == 2


def test_sequential_iter():
    model = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)))
    assert list(model) == [model[0], model[1]]


def test_sequential_reversed():
    model = Sequential((Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)))
    assert list(reversed(model)) == [model[1], model[0]]
