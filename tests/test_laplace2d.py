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

import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import Laplace2D


def test_laplace2d():
    x = Laplace2D()(jnp.arange(1, 26).reshape([1, 5, 5]).astype(jnp.float32))

    y = jnp.array(
        [
            [
                [4.0, 3.0, 2.0, 1.0, -6.0],
                [-5.0, 0.0, 0.0, 0.0, -11.0],
                [-10.0, 0.0, 0.0, 0.0, -16.0],
                [-15.0, 0.0, 0.0, 0.0, -21.0],
                [-46.0, -27.0, -28.0, -29.0, -56.0],
            ]
        ]
    )

    npt.assert_allclose(x, y)
