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

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy.testing as npt

from serket.nn import AdjustContrast2D, RandomContrast2D


def test_adjust_contrast_2d():
    x = jnp.array(
        [
            [
                [0.19165385, 0.4459561, 0.03873193],
                [0.58923364, 0.0923605, 0.2597469],
                [0.83097064, 0.4854728, 0.03308535],
            ],
            [
                [0.10485303, 0.10068893, 0.408355],
                [0.40298176, 0.6227188, 0.8612417],
                [0.52223504, 0.3363577, 0.1300546],
            ],
        ]
    )
    y = jnp.array(
        [
            [
                [0.26067203, 0.38782316, 0.18421106],
                [0.45946193, 0.21102534, 0.29471856],
                [0.5803304, 0.4075815, 0.18138777],
            ],
            [
                [0.24628687, 0.24420482, 0.39803785],
                [0.39535123, 0.50521976, 0.6244812],
                [0.45497787, 0.3620392, 0.25888765],
            ],
        ]
    )

    npt.assert_allclose(AdjustContrast2D(contrast_factor=0.5)(x), y, atol=1e-5)


def test_random_contrast_2d():
    x = jnp.array(
        [
            [
                [0.19165385, 0.4459561, 0.03873193],
                [0.58923364, 0.0923605, 0.2597469],
                [0.83097064, 0.4854728, 0.03308535],
            ],
            [
                [0.10485303, 0.10068893, 0.408355],
                [0.40298176, 0.6227188, 0.8612417],
                [0.52223504, 0.3363577, 0.1300546],
            ],
        ]
    )

    y = jnp.array(
        [
            [
                [0.23179087, 0.4121493, 0.1233343],
                [0.5137658, 0.1613692, 0.28008443],
                [0.68521255, 0.44017565, 0.11932957],
            ],
            [
                [0.18710288, 0.1841496, 0.40235513],
                [0.39854428, 0.55438805, 0.7235553],
                [0.4831221, 0.3512926, 0.20497654],
            ],
        ]
    )

    npt.assert_allclose(
        RandomContrast2D(contrast_range=(0.5, 1))(x, key=jax.random.PRNGKey(0)),
        y,
        atol=1e-5,
    )
