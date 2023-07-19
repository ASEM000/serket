# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import jax
import jax.random as jr
import jax.tree_util as jtu
import pytest

from serket.nn.initialization import resolve_init_func
from serket.nn.utils import canonicalize


def test_canonicalize_init_func():
    k = jr.PRNGKey(0)

    assert resolve_init_func("he_normal")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("he_uniform")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("glorot_normal")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("glorot_uniform")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("lecun_normal")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("lecun_uniform")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("normal")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("uniform")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("ones")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("zeros")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("xavier_normal")(k, (2, 2)).shape == (2, 2)
    assert resolve_init_func("xavier_uniform")(k, (2, 2)).shape == (2, 2)

    assert isinstance(resolve_init_func(jax.nn.initializers.he_normal()), jtu.Partial)
    assert isinstance(resolve_init_func(None), jtu.Partial)

    with pytest.raises(ValueError):
        resolve_init_func("invalid")

    with pytest.raises(ValueError):
        resolve_init_func(1)


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
