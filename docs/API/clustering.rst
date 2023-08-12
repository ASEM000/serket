Clustering
---------------------------------
.. currentmodule:: serket.nn

    
.. autoclass:: KMeans
    :members:
        __call__

.. note::

    Example usage plot of :class:`.nn.KMeans`

    .. code-block::

        >>> import jax
        >>> import jax.random as jr
        >>> import matplotlib.pyplot as plt
        >>> import serket as sk
        >>> x = jr.uniform(jr.PRNGKey(0), shape=(500, 2))
        >>> layer = sk.nn.KMeans(clusters=5, tol=1e-6)
        >>> x, state = layer(x)
        >>> plt.scatter(x[:, 0], x[:, 1], c=state.labels[:, 0], cmap="jet_r")
        >>> plt.scatter(state.centers[:, 0], state.centers[:, 1], c="r", marker="o", linewidths=4)
    .. image:: kmeans.svg
        :width: 600
        :align: center