from serket.nn import Lambda, Sequential


def test_sequential():

    model = Sequential([Lambda(lambda x: x)])
    assert model(1.0) == 1.0

    model = Sequential([Lambda(lambda x: x + 1), Lambda(lambda x: x + 1)])
    assert model(1.0) == 3.0

    model = Sequential([lambda x: x])
    assert model(1.0) == 1.0
