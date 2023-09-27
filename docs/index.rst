


|logo| Serket
==============

- ``serket`` aims to be the most intuitive and easy-to-use neural network library in ``jax``.
- ``serket`` is fully transparent to ``jax`` transformation (e.g. ``vmap``, ``grad``, ``jit``,...).

.. |logo| image:: _static/logo.svg
    :height: 40px    




🛠️ Installation
----------------

Install from github::

   pip install git+https://github.com/ASEM000/serket


🏃 Quick example
------------------

.. code-block:: python

    import jax, jax.numpy as jnp
    import serket as sk

    x_train, y_train = ..., ...
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

    net = sk.nn.Sequential(
        jnp.ravel,
        sk.nn.Linear(28 * 28, 64, key=k1),
        jax.nn.relu,
        sk.nn.Linear(64, 64, key=k2),
        jax.nn.relu,
        sk.nn.Linear(64, 10, key=k3),
    )

    net = sk.tree_mask(net)

    def softmax_cross_entropy(logits, onehot):
        return -jnp.sum(labels * jax.nn.log_softmax(logits, axis=-1), axis=-1)

    def update(param, grad):
        return param - grad * 1e-3

    @ft.partial(jax.grad, has_aux=True)
    def loss_func(net, x, y):
        net = sk.tree_unmask(net)
        logits = jax.vmap(net)(x)
        onehot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(softmax_cross_entropy(logits, onehot))
        return loss, (loss, logits)

    @jax.jit
    def train_step(net, x, y):
        grads, (loss, logits) = loss_func(net, x, y)
        net = jax.tree_map(update, net, grads)
        return net, (loss, logits)

    for j, (xb, yb) in enumerate(zip(x_train, y_train)):
        net, (loss, logits) = train_step(net, xb, yb)
        accuracy = accuracy_func(logits, y_train)

    net = sk.tree_unmask(net)


.. toctree::
    :caption: 👋 Introduction
    :maxdepth: 1
    
    notebooks/mental_model

.. toctree::
    :caption: 📖 Guides
    :maxdepth: 1
    
    train_examples
    interoperability
    notebooks/evaluation
    notebooks/mixed_precision
    notebooks/checkpointing
    notebooks/subset_training
    notebooks/mini_optax
    notebooks/common_recipes


.. currentmodule:: serket
    


.. toctree::
    :caption: 📃 API Documentation
    :maxdepth: 1
    
    notebooks/layers_overview
    API/common
    API/cluster
    API/nn
    API/image
    API/sepes

Apache2.0 License.

Indices
=======

* :ref:`genindex`


