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

from typing import Any

import jax
import jax.random as jr
from jax.lax import stop_gradient

import serket as sk
from serket.nn.reshape import Pad2D, RandomCrop2D, Resize2D
from serket.nn.utils import Range


@sk.autoinit
class RandomApply(sk.TreeClass):
    """
    Randomly applies a layer with probability p.

    Args:
        layer: layer to apply.
        p: probability of applying the layer

    Example:
        >>> import serket as sk
        >>> import jax.numpy as jnp
        >>> layer = sk.nn.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=0.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 10, 10)
        >>> layer = sk.nn.RandomApply(sk.nn.MaxPool2D(kernel_size=2, strides=2), p=1.0)
        >>> layer(jnp.ones((1, 10, 10))).shape
        (1, 5, 5)

    Reference:
        - https://pytorch.org/vision/main/_modules/torchvision/transforms/transforms.html#RandomApply
        - Use :func:`nn.Sequential` to apply multiple layers.
    """

    layer: Any
    p: float = sk.field(default=0.5, callbacks=[Range(0, 1)])

    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)):
        if not jr.bernoulli(key, stop_gradient(self.p)):
            return x
        return self.layer(x)


class RandomZoom2D(sk.TreeClass):
    def __init__(
        self,
        height_factor: tuple[float, float] = (0.0, 1.0),
        width_factor: tuple[float, float] = (0.0, 1.0),
    ):
        """Randomly zooms an image.

        Positive values are zoom in, negative values are zoom out.

        Args:
            height_factor: (min, max)
            width_factor: (min, max)

        Reference:
            - https://www.tensorflow.org/api_docs/python/tf/keras/layers/RandomZoom
        """
        if not (isinstance(height_factor, tuple) and len(height_factor) == 2):
            raise ValueError("height_factor must be a tuple of length 2")

        if not (isinstance(width_factor, tuple) and len(width_factor) == 2):
            raise ValueError("width_factor must be a tuple of length 2")

        self.height_factor = height_factor
        self.width_factor = width_factor

    def __call__(self, x: jax.Array, key: jr.KeyArray = jr.PRNGKey(0)) -> jax.Array:
        keys = jr.split(key, 4)

        height_factor = jax.lax.stop_gradient(self.height_factor)
        width_factor = jax.lax.stop_gradient(self.width_factor)

        height_factor = jr.uniform(
            keys[0],
            shape=(),
            minval=height_factor[0],
            maxval=height_factor[1],
        )
        width_factor = jr.uniform(
            keys[1],
            shape=(),
            minval=width_factor[0],
            maxval=width_factor[1],
        )

        R, C = x.shape[1:3]  # R = rows, C = cols
        RR, CC = int(R * (1 + height_factor))  # RR = resized rows,
        CC = int(C * (1 + width_factor))  # CC = resized cols

        if height_factor > 0:
            # zoom in rows
            x = Resize2D((RR, C))(x)
            x = RandomCrop2D((R, C))(x, key=keys[2])

        if width_factor > 0:
            # zoom in cols
            x = Resize2D((R, CC))(x)
            x = RandomCrop2D((R, C))(x, key=keys[3])

        if height_factor < 0:
            # zoom out rows
            x = Resize2D((RR, C))(x)
            x = Pad2D((((R - RR) // 2, (R - RR) - ((R - RR) // 2)), (0, 0)))(x)

        if width_factor < 0:
            # zoom out cols
            x = Resize2D((R, CC))(x)
            x = Pad2D(((0, 0), ((C - CC) // 2, (C - CC) - (C - CC) // 2)))(x)

        return x

    @property
    def spatial_ndim(self) -> int:
        return 2
