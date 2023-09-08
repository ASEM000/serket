


|logo| Serket
==============

- ``serket`` aims to be the most intuitive and easy-to-use neural network library in ``jax``.
- ``serket`` is fully transparent to ``jax`` transformation (e.g. ``vmap``, ``grad``, ``jit``,...).

.. |logo| image:: _static/kol.svg
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
    import optax

    x_train, y_train = ..., ...
    k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

    nn = sk.nn.Sequential(
        sk.nn.Linear(28 * 28, 64, key=k1), jax.nn.relu,
        sk.nn.Linear(64, 64, key=k2), jax.nn.relu,
        sk.nn.Linear(64, 10, key=k3),
    )

    nn = sk.tree_mask(nn)  # pass non-jaxtype through jax-transforms
    optim = optax.adam(LR)
    optim_state = optim.init(nn)

    @ft.partial(jax.grad, has_aux=True)
    def loss_func(nn, x, y):
        nn = sk.tree_unmask(nn)
        logits = jax.vmap(nn)(x)
        onehot = jax.nn.one_hot(y, 10)
        loss = jnp.mean(optax.softmax_cross_entropy(logits, onehot))
        return loss, (loss, logits)

    @jax.jit
    def train_step(nn, optim_state, x, y):
        grads, (loss, logits) = loss_func(nn, x, y)
        updates, optim_state = optim.update(grads, optim_state)
        nn = optax.apply_updates(nn, updates)
        return nn, optim_state, (loss, logits)

    for j, (xb, yb) in enumerate(zip(x_train, y_train)):
        nn, optim_state, (loss, logits) = train_step(nn, optim_state, xb, yb)
        accuracy = accuracy_func(logits, y_train)

    nn = sk.tree_unmask(nn)


.. toctree::
    :caption: Introduction
    :maxdepth: 1
    
    notebooks/mental_model
    notebooks/train_eval
    notebooks/layers_overview
    notebooks/lazy_initialization

.. toctree::
    :caption: Examples
    :maxdepth: 1
    
    examples


.. currentmodule:: serket
    


.. toctree::
    :caption: API Documentation
    :maxdepth: 1
    
    API/common
    API/cluster
    API/nn
    API/image
    API/pytreeclass

Apache2.0 License.

Indices
=======

* :ref:`genindex`


