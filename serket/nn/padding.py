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

import abc
import functools as ft

import jax
import jax.numpy as jnp
import pytreeclass as pytc

from serket.nn.utils import delayed_canonicalize_padding, validate_spatial_in_shape


class PadND(pytc.TreeClass):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        """
        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        self.padding = delayed_canonicalize_padding(
            in_dim=None,
            padding=padding,
            kernel_size=((1,),) * self.spatial_ndim,
            strides=None,
        )
        self.value = value

    @ft.partial(validate_spatial_in_shape, attribute_name="spatial_ndim")
    def __call__(self, x: jax.Array, **k) -> jax.Array:
        # do not pad the channel axis
        shape = ((0, 0), *self.padding)
        return jax.lax.stop_gradient(jnp.pad(x, shape, constant_values=self.value))

    @property
    @abc.abstractmethod
    def spatial_ndim(self) -> int:
        """Number of spatial dimensions of the image."""
        ...


class Pad1D(PadND):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        """
        Pad a 1D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(padding=padding, value=value)

    @property
    def spatial_ndim(self) -> int:
        return 1


class Pad2D(PadND):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        """
        Pad a 2D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(padding=padding, value=value)

    @property
    def spatial_ndim(self) -> int:
        return 2


class Pad3D(PadND):
    def __init__(self, padding: int | tuple[int, int], value: float = 0.0):
        """
        Pad a 3D tensor.

        Args:
            padding: padding to apply to each side of the input.
            value: value to pad with. Defaults to 0.0.

        Note:
            https://jax.readthedocs.io/en/latest/_autosummary/jax.numpy.pad.html
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding1D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding2D
            https://www.tensorflow.org/api_docs/python/tf/keras/layers/ZeroPadding3D
        """
        super().__init__(padding=padding, value=value)

    @property
    def spatial_ndim(self) -> int:
        return 3
