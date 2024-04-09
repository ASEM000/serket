# Copyright 2024 serket authors
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

# grayscale

from __future__ import annotations

import functools as ft
from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
from typing_extensions import Annotated

from serket import TreeClass
from serket._src.utils.convert import canonicalize
from serket._src.utils.typing import CHWArray, HWArray
from serket._src.utils.validate import validate_spatial_ndim


def rgb_to_grayscale(image: CHWArray, weights: jax.Array | None = None) -> CHWArray:
    """Converts an RGB image to grayscale.

    Args:
        image: RGB image.
        weights: Weights for each channel.
    """
    c, _, _ = image.shape
    assert c == 3

    if weights is None:
        weights = jnp.array([76, 150, 29]) / (1 if image.dtype == jnp.uint8 else 255.0)

    rw, gw, bw = weights
    r, g, b = jnp.split(image, 3, axis=0)
    return rw * r + gw * g + bw * b


def grayscale_to_rgb(image: CHWArray) -> CHWArray:
    """Converts a single channel image to RGB."""
    c, _, _ = image.shape
    assert c == 1
    return jnp.concatenate([image, image, image], axis=0)


def rgb_to_hsv(image: CHWArray) -> CHWArray:
    """Convert an RGB image to HSV.

    Args:
        image: RGB image in channel-first format with range [0, 1].

    Returns:
        HSV image in channel-first format with range [0, 2pi] for hue, [0, 1] for saturation and value.
    """
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv

    eps = jnp.finfo(image.dtype).eps
    maxc = jnp.max(image, axis=0, keepdims=True)
    argmaxc = jnp.argmax(image, axis=0, keepdims=True)
    minc = jnp.min(image, axis=0, keepdims=True)
    diff = maxc - minc

    diff = jnp.where(diff == 0, 1, diff)
    rc, gc, bc = jnp.split((maxc - image), 3, axis=0)

    h1 = bc - gc
    h2 = (rc - bc) + 2.0 * diff
    h3 = (gc - rc) + 4.0 * diff

    h = jnp.stack((h1, h2, h3), axis=0) / (diff + eps)
    h = jnp.take_along_axis(h, argmaxc[None], axis=0).squeeze(0)
    h = (h / 6.0) % 1.0
    h = 2.0 * jnp.pi * h
    # saturation
    s = diff / (maxc + eps)
    # value
    v = maxc
    return jnp.concatenate((h, s, v), axis=0)


def hsv_to_rgb(image: CHWArray) -> CHWArray:
    """Convert an image from HSV to RGB."""
    # https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html#rgb_to_hsv
    c, _, _ = image.shape
    assert c == 3

    h = image[0] / (2 * jnp.pi)
    s = image[1]
    v = image[2]

    hi = jnp.floor(h * 6) % 6
    f = ((h * 6) % 6) - hi
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    indices = jnp.stack([hi, hi + 6, hi + 12], axis=0).astype(jnp.int32)
    out = jnp.stack((v, q, p, p, t, v, t, v, v, q, p, p, p, p, t, v, v, q))
    out = jnp.take_along_axis(out, indices, axis=0)
    return out


class KMeansState(NamedTuple):
    centers: Annotated[jax.Array, "Float[k,d]"]
    error: Annotated[jax.Array, "Float[k,d]"]
    iters: int = 0


def distances_from_centers(
    data: Annotated[jax.Array, "Float[n,d]"],
    centers: Annotated[jax.Array, "Float[k,d]"],
) -> Annotated[jax.Array, "Float[n,k]"]:
    # for each point find the distance to each center
    return jax.vmap(lambda xi: jax.vmap(jnp.linalg.norm)(xi - centers))(data)


def labels_from_distances(
    distances: Annotated[jax.Array, "Float[n,k]"]
) -> Annotated[jax.Array, "Integer[n,1]"]:
    # for each point find the index of the closest center
    return jnp.argmin(distances, axis=1, keepdims=True)


def centers_from_labels(
    data: Annotated[jax.Array, "Float[n,d]"],
    labels: Annotated[jax.Array, "Integer[n,1]"],
    k: int,
) -> Annotated[jax.Array, "Float[k,d]"]:
    # for each center find the mean of the points assigned to it
    return jax.vmap(
        lambda k: jnp.divide(
            jnp.sum(jnp.where(labels == k, data, 0), axis=0),
            jnp.sum(jnp.where(labels == k, 1, 0)).clip(min=1),
        )
    )(jnp.arange(k))


def kmeans(
    data: Annotated[jax.Array, "Float[n,d]"],
    state: KMeansState,
    *,
    clusters: int,
    tol: float = 1e-4,
):
    def step(state: KMeansState) -> KMeansState:
        # Float[n,d] -> Float[n,k]
        distances = distances_from_centers(data, state.centers)
        # Float[n,k] -> Integer[n,1]
        labels = labels_from_distances(distances)
        centers = centers_from_labels(data, labels, clusters)
        error = jnp.abs(centers - state.centers)
        return KMeansState(centers, error, state.iters + 1)

    def condition(state: KMeansState) -> bool:
        return jnp.all(state.error > tol)

    return jax.lax.while_loop(condition, step, state)


def kmeans_color_quantization_2d(
    image: HWArray,
    k: int,
    *,
    key: jax.Array,
    tol: float = 1e-4,
):
    """Channel K-means color quantization.

    Args:
        image: 2D image.
        k: The number of colors to quantize to for each channel.
        key: Random key to initialize the centers.
        tol: The tolerance for convergence. default: 1e-4
    """
    assert image.ndim == 2
    flat_image = image.reshape(-1, 1)
    minval, maxval = jnp.min(flat_image), jnp.max(flat_image)
    centers = jax.random.uniform(key, (k, 1), float, minval, maxval)
    state = KMeansState(centers, centers + jnp.inf, 0)
    state = kmeans(flat_image, state, clusters=k, tol=tol)
    distances = distances_from_centers(flat_image, state.centers)
    labels = labels_from_distances(distances)
    return state.centers[labels].reshape(image.shape)


class RGBToGrayscale2D(TreeClass):
    """Converts a channel-first RGB image to grayscale.

    .. image:: ../_static/rgbtograyscale2d.png

    Args:
        weights: Weights for each channel.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> rgb_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.RGBToGrayscale2D()
        >>> gray_image = layer(rgb_image)
        >>> gray_image.shape
        (1, 5, 5)
    """

    def __init__(self, weights: jax.Array | None = None):
        self.weights = weights

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return rgb_to_grayscale(image, self.weights)

    spatial_ndim: int = 2


class GrayscaleToRGB2D(TreeClass):
    """Converts a grayscale image to RGB.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> gray_image = jnp.ones([1, 5, 5])
        >>> layer = sk.image.GrayscaleToRGB2D()
        >>> rgb_image = layer(gray_image)
        >>> rgb_image.shape
        (3, 5, 5)
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return grayscale_to_rgb(image)

    spatial_ndim: int = 2


class RGBToHSV2D(TreeClass):
    """Converts an RGB image to HSV.

    .. image:: ../_static/rgbtohsv2d.png

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> rgb_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.RGBToHSV2D()
        >>> hsv_image = layer(rgb_image)
        >>> hsv_image.shape
        (3, 5, 5)

    Reference:
        - https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return rgb_to_hsv(image)

    spatial_ndim: int = 2


class HSVToRGB2D(TreeClass):
    """Converts an HSV image to RGB.

    Example:
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> hsv_image = jnp.ones([3, 5, 5])
        >>> layer = sk.image.HSVToRGB2D()
        >>> rgb_image = layer(hsv_image)
        >>> rgb_image.shape
        (3, 5, 5)

    Reference:
        - https://kornia.readthedocs.io/en/latest/_modules/kornia/color/hsv.html
    """

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray) -> CHWArray:
        return hsv_to_rgb(image)

    spatial_ndim: int = 2


class KMeansColorQuantization2D(TreeClass):
    """Channel-wise K-means color quantization.

    .. image:: ../_static/kmeans_quantization.png
        :width: 600
        :align: center

    Args:
        k: The number of colors to quantize to for each channel. Accepts a single
            integer or a sequence of integers corresponding to the number of
            clusters for each channel.
        tol: The tolerance for convergence. default: 1e-4

    Example:
        Quanitze an image to 2, 3, and 4 colors for the red, green, and blue channels
        respectively.

        >>> import jax
        >>> import jax.random as jr
        >>> import jax.numpy as jnp
        >>> import serket as sk
        >>> image = jnp.ones([3, 5, 5])
        >>> k1, k2 = jr.split(jr.PRNGKey(0))
        >>> layer = sk.image.KMeansColorQuantization2D(k=[2, 3, 4])
        >>> image = jr.uniform(k1, shape=(3, 50, 50))
        >>> quantized_image = layer(image, key=k2)
        >>> r_quantized, g_quantized, b_quantized = quantized_image
        >>> assert len(jnp.unique(r_quantized)) == 2
        >>> assert len(jnp.unique(g_quantized)) == 3
        >>> assert len(jnp.unique(b_quantized)) == 4
    """

    def __init__(self, k: int | Sequence[int], tol: float = 1e-4):
        self.k = canonicalize(k, ndim=3, name="k")
        self.tol = tol

    @ft.partial(validate_spatial_ndim, argnum=0)
    def __call__(self, image: CHWArray, *, key: jax.Array) -> CHWArray:
        tol = jax.lax.stop_gradient(self.tol)
        k = jax.lax.stop_gradient(self.k)
        r_k, g_k, b_k = k
        r_key, g_key, b_key = jax.random.split(key, 3)
        r_image, g_image, b_image = image
        qunatize = ft.partial(kmeans_color_quantization_2d, tol=tol)
        r_image_q = qunatize(r_image, r_k, key=r_key)
        g_image_q = qunatize(g_image, g_k, key=g_key)
        b_image_q = qunatize(b_image, b_k, key=b_key)
        return jnp.stack([r_image_q, g_image_q, b_image_q])

    spatial_ndim: int = 2
