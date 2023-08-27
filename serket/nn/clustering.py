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

import functools as ft
from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from typing_extensions import Annotated

import serket as sk
from serket.nn.custom_transform import tree_eval, tree_state
from serket.nn.utils import IsInstance, Range

"""K-means utility functions."""


class KMeansState(NamedTuple):
    centers: Annotated[jax.Array, "f32[k,d]"]
    error: Annotated[jax.Array, "f32[k,d]"]
    iters: int = 0


def distances_from_centers(
    data: Annotated[jax.Array, "f32[n,d]"],
    centers: Annotated[jax.Array, "f32[k,d]"],
) -> Annotated[jax.Array, "f32[n,k]"]:
    # for each point find the distance to each center
    return jax.vmap(lambda xi: jax.vmap(jnp.linalg.norm)(xi - centers))(data)


def labels_from_distances(
    distances: Annotated[jax.Array, "f32[n,k]"]
) -> Annotated[jax.Array, "f32[n,1]"]:
    # for each point find the index of the closest center
    return jnp.argmin(distances, axis=1, keepdims=True)


def centers_from_labels(
    data: Annotated[jax.Array, "f32[n,d]"],
    labels: Annotated[jax.Array, "i32[n,1]"],
    k: int,
) -> Annotated[jax.Array, "f32[k,d]"]:
    # for each center find the mean of the points assigned to it
    return jax.vmap(
        lambda k: jnp.divide(
            jnp.sum(jnp.where(labels == k, data, 0), axis=0),
            jnp.sum(jnp.where(labels == k, 1, 0)).clip(min=1),
        )
    )(jnp.arange(k))


@ft.partial(jax.jit, static_argnames="clusters")
def kmeans(
    data: Annotated[jax.Array, "f32[n,d]"],
    state: KMeansState,
    *,
    clusters: int,
    tol: float = 1e-4,
) -> KMeansState:
    """K-means clustering algorithm.

    Steps:
        1. Initialize the centers randomly. f32[k,d]
        2. Calculate point-wise distances from data and centers. f32[n,d],f32[k,d] -> f32[n,k]
        3. Assign each point to the closest center. f32[n,k] -> f32[n,1]
        4. Calculate the new centers from data and labels. f32[n,d],f32[n,1] -> f32[k,d]
        5. Repeat steps 2-4 until the centers converge.

    Args:
        data: The data to cluster in the shape of n points with d dimensions.
        state: initial ``KMeansState`` containing:

            - centers: The initial centers of the clusters.
            - error: The initial error of the centers at each iteration.
            - iters: The inital number of iterations (i.e. 0)

        clusters: The number of clusters.
        tol: The tolerance for convergence. default: 1e-4

    Returns:
        A ``KMeansState`` named tuple containing:

            - centers: The final centers of the clusters.
            - error: The error of the centers at each iteration.
            - iters: The number of iterations until convergence.
    """

    if not isinstance(state, KMeansState):
        raise TypeError(f"{state=} not an instance of `KMeansState`")

    def step(state: KMeansState) -> KMeansState:
        # f32[n,d] -> f32[n,k]
        distances = distances_from_centers(data, state.centers)

        # f32[n,k] -> f32[n,1]
        labels = labels_from_distances(distances)

        # f32[n,d] -> f32[k,d]
        centers = centers_from_labels(data, labels, clusters)

        error = jnp.abs(centers - state.centers)

        return KMeansState(centers, error, state.iters + 1)

    def condition(state: KMeansState) -> bool:
        return jnp.all(state.error > tol)

    return jax.lax.while_loop(condition, step, state)


@sk.autoinit
class KMeans(sk.TreeClass):
    """Vanilla K-means clustering algorithm.

    Args:
        clusters: The number of clusters.
        tol: The tolerance for convergence. default: 1e-4

    Example:

        Example usage plot of :class:`.nn.KMeans`

        >>> import jax
        >>> import jax.random as jr
        >>> import matplotlib.pyplot as plt
        >>> import serket as sk
        >>> x = jr.uniform(jr.PRNGKey(0), shape=(500, 2))
        >>> layer = sk.nn.KMeans(clusters=5, tol=1e-6)
        >>> labels, state = layer(x)
        >>> plt.scatter(x[:, 0], x[:, 1], c=labels[:, 0], cmap="jet_r")  # doctest: +SKIP
        >>> plt.scatter(state.centers[:, 0], state.centers[:, 1], c="r", marker="o", linewidths=4)  # doctest: +SKIP
        
        .. image:: ../_static/kmeans.svg
            :width: 600
            :align: center

    Example:
        >>> import serket as sk
        >>> import jax.random as jr
        >>> features = 3
        >>> clusters = 4
        >>> x = jr.uniform(jr.PRNGKey(0), shape=(100, features))
        >>> layer = sk.nn.KMeans(clusters=clusters, tol=1e-6)
        >>> labels, state = layer(x)
        >>> centers = state.centers
        >>> assert labels.shape == (100, 1)
        >>> assert centers.shape == (clusters, features)

    Note:
        To use the :class:`.nn.KMeans` layer in evaluation mode, use :func:`.tree_eval` to
        disallow centers update and only predict the labels based on the current
        centers.

        >>> import serket as sk
        >>> import jax.random as jr
        >>> features = 3
        >>> clusters = 4
        >>> x = jr.uniform(jr.PRNGKey(0), shape=(100, features))
        >>> layer = sk.nn.KMeans(clusters=clusters, tol=1e-6)
        >>> x, state = layer(x)
        >>> eval_layer = sk.tree_eval(layer)
        >>> y = jr.uniform(jr.PRNGKey(0), shape=(1, features))
        >>> y, eval_state = eval_layer(y, state)
        >>> # centers are not updated
        >>> assert jnp.all(eval_state.centers == state.centers)
    """

    clusters: int = sk.field(callbacks=[IsInstance(int), Range(1)])
    tol: float = sk.field(callbacks=[IsInstance(float), Range(0, min_inclusive=False)])

    def __call__(
        self,
        x: jax.Array,
        state: KMeansState | None = None,
    ) -> tuple[jax.Array, KMeansState]:
        """K-means clustering algorithm.

        Args:
            x: The data to cluster in the shape of n points with d dimensions.
            state: initial ``KMeansState`` containing:

                - centers: The initial centers of the clusters.
                - error: The initial error of the centers at each iteration.
                - iters: The inital number of iterations (i.e. 0)

                if ``None`` then the initial state is initialized using the rule
                defined in :func:`.tree_state`

        Returns:
            A tuple containing the labels and a ``KMeansState``.
        """

        state = sk.tree_state(self, array=x) if state is None else state
        clusters, tol, state = jax.lax.stop_gradient((self.clusters, self.tol, state))
        state = kmeans(x, state, clusters=clusters, tol=tol)
        distances = distances_from_centers(x, state.centers)
        labels = labels_from_distances(distances)
        return labels, state


class EvalKMeans(sk.TreeClass):
    """K-means clustering algorithm evaluation.

    Evaluates the K-means clustering algorithm on the input data and returns the
    input data and the final ``KMeansState`` with no further updates.
    """

    def __call__(
        self,
        x: jax.Array,
        state: KMeansState,
    ) -> tuple[jax.Array, KMeansState]:
        distances = distances_from_centers(x, state.centers)
        labels = labels_from_distances(distances)
        state = state._replace(iters=None, error=None)
        return labels, state


@tree_state.def_state(KMeans)
def init_kmeans(layer: KMeans, *, array: jax.Array) -> KMeansState:
    centers = jr.uniform(
        key=jr.PRNGKey(0),
        minval=array.min(),
        maxval=array.max(),
        shape=(layer.clusters, array.shape[1]),
    )

    return KMeansState(centers=centers, error=centers + jnp.inf, iters=0)


@tree_eval.def_eval(KMeans)
def eval_kmeans(_: KMeans) -> EvalKMeans:
    return EvalKMeans()
