# Copyright 2023 serket authors
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


import warnings

import jax.numpy as jnp
import numpy.testing as npt
from jax import random

import serket as sk

# Suppress FutureWarning
warnings.simplefilter(action="ignore", category=FutureWarning)


def test_kmeans():
    from sklearn.cluster import KMeans

    rng = random.PRNGKey(42)
    k = 3
    x = random.uniform(rng, (100, 2))
    sc_ = KMeans(n_clusters=k, tol=1e-5).fit(x)
    layer = sk.cluster.KMeans(k, tol=1e-5)
    _, state = layer(x)
    npt.assert_allclose(
        jnp.sort(sc_.cluster_centers_, axis=0),
        jnp.sort(state.centers, axis=0),
        atol=1e-6,
    )
    # pick a point near one of the centers
    xx = jnp.array([[0.5, 0.2]])
    labels, eval_state = sk.tree_eval(layer)(xx, state)
    # centers should not change
    npt.assert_allclose(state.centers, eval_state.centers, atol=1e-6)
    assert labels[0] == 0
