<div align="center">
<img width="250px" src="assets/logo.svg"></div>

<h2 align="center">The ✨Magical✨ JAX ML Library.</h2>
<h5 align = "center"> *Serket is the goddess of magic in Egyptian mythology

[**Installation**](#Installation)
|[**Description**](#Description)
|[**Quick Example**](#QuickExample)

![Tests](https://github.com/ASEM000/serket/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.8%203.9%203.10%203.11-red)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![Downloads](https://pepy.tech/badge/serket)](https://pepy.tech/project/serket)
[![codecov](https://codecov.io/gh/ASEM000/serket/branch/main/graph/badge.svg?token=C6NXOK9EVS)](https://codecov.io/gh/ASEM000/serket)
[![Documentation Status](https://readthedocs.org/projects/serket/badge/?version=latest)](https://serket.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/526985786.svg)](https://zenodo.org/badge/latestdoi/526985786)
![PyPI](https://img.shields.io/pypi/v/serket)
[![CodeFactor](https://www.codefactor.io/repository/github/asem000/serket/badge)](https://www.codefactor.io/repository/github/asem000/serket)

</h5>

## 🛠️ Installation<a id="Installation"></a>

**Install development version**

```python
pip install git+https://github.com/ASEM000/serket
```

## 📖 Description and motivation<a id="Description"></a>

- `serket` aims to be the most intuitive and easy-to-use neural network library in `JAX`.
- `serket` is fully transparent to `jax` transformation (e.g. `vmap`,`grad`,`jit`,...).

### 🏃 Quick example<a id="QuickExample"></a>

For full examples see [here](https://serket.readthedocs.io/en/latest/examples.html) e.g. [Training 🚆 MNIST](https://serket.readthedocs.io/en/latest/notebooks/mnist.html), or [Training 🚆 Bidirectional-LSTM](https://serket.readthedocs.io/en/latest/notebooks/bilstm.html)

```python
import jax, jax.numpy as jnp
import serket as sk
import optax

x_train, y_train = ..., ...  # samples, 1, 28, 28
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

net = sk.nn.Sequential(
    jnp.ravel,
    sk.nn.Linear(28 * 28, 64, key=k1),
    jax.nn.relu,
    sk.nn.Linear(64, 64, key=k2),
    jax.nn.relu,
    sk.nn.Linear(64, 10, key=k3),
)

net = sk.tree_mask(net)  # pass non-jaxtype through jax-transforms
optim = optax.adam(LR)
optim_state = optim.init(net)

@ft.partial(jax.grad, has_aux=True)
def loss_func(net, x, y):
    net = sk.tree_unmask(net)
    logits = jax.vmap(net)(x)
    onehot = jax.nn.one_hot(y, 10)
    loss = jnp.mean(optax.softmax_cross_entropy(logits, onehot))
    return loss, (loss, logits)

@jax.jit
def train_step(net, optim_state, x, y):
    grads, (loss, logits) = loss_func(net, x, y)
    updates, optim_state = optim.update(grads, optim_state)
    net = optax.apply_updates(net, updates)
    return net, optim_state, (loss, logits)

for j, (xb, yb) in enumerate(zip(x_train, y_train)):
    net, optim_state, (loss, logits) = train_step(net, optim_state, xb, yb)
    accuracy = accuracy_func(logits, y_train)

net = sk.tree_unmask(net)
```

#### Notable features:

<details><summary>🥱 Functional lazy initialization </summary>

Lazy initialization is particularly useful in scenarios where the dimensions of certain input features are not known in advance. For instance, consider a situation where the number of neurons required for a flattened image input is uncertain (**Example 1**), or the shape of the output from a flattened convolutional layer is not straightforward to calculate (**Example 2**). In such cases, lazy initialization allows the model to defer the allocation of memory for these uncertain dimensions until they are explicitly computed during the training process. This flexibility ensures that the model can handle varying input sizes and adapt its architecture accordingly, making it more versatile and efficient when dealing with different data samples or changing conditions.

_Example 1_

```python
import jax
import serket as sk

# 10 images from MNIST
x = jax.numpy.ones([5, 1, 28, 28])

layer = sk.nn.Sequential(
    jax.numpy.ravel,
    # lazy in_features inference pass `None`
    sk.nn.Linear(None, 10),
    jax.nn.relu,
    sk.nn.Linear(10, 10),
    jax.nn.softmax,
)
# materialize the layer with single image
_, layer = layer.at["__call__"](x[0])
# apply on batch
y = jax.vmap(layer)(x)
y.shape
(5, 10)
```

_Example 2_

```python
import jax
import serket as sk

# 10 images from MNIST
x = jax.numpy.ones([5, 1, 28, 28])

layer = sk.nn.Sequential(
    sk.nn.Conv2D(1, 10, 3),
    jax.nn.relu,
    sk.nn.MaxPool2D(2),
    jax.numpy.ravel,
    # linear input size is inferred from
    # previous layer output
    sk.nn.Linear(None, 10),
    jax.nn.softmax,
)

# materialize the layer with single image
_, layer = layer.at["__call__"](x[0])

# apply on batch
y = jax.vmap(layer)(x)

y.shape
# (5, 10)
```

</details>

<!-- <details><summary>Evaluation behavior handling</summary>

`serket` uses `functools` dispatching to modifiy a tree of layers to disable any training-related behavior during evaluation. It replaces certain layers, such as `Dropout` and `BatchNorm`, with equivalent layers that don't affect the model's output during evaluation.

for example:

```python
# dropout is replaced by an identity layer in evaluation mode
# by registering `tree_eval.def_eval(sk.nn.Dropout, sk.nn.Identity)`
import jax.numpy as jnp
import serket as sk
layer = sk.nn.Dropout(0.5)
sk.tree_eval(layer)
# Identity()
```

Let's break down the code snippet and its purpose:

1. The function `tree_eval(tree)` takes a tree of layers as input.
2. The function replaces specific layers in the tree with evaluation-specific layers.

Here are the modifications it makes to the tree:

- If a `Dropout` layer is encountered in the tree, it is replaced with an `Identity` layer. The `Identity` layer is a simple layer that doesn't introduce any changes to the input, making it effectively a no-op during evaluation.

- If a `BatchNorm` layer is encountered in the tree, it is replaced with an `EvalNorm` layer. The `EvalNorm` layer is designed to have the same behavior as `BatchNorm` during evaluation but not during training.

The purpose of these replacements is to ensure that the evaluation behavior is part of the


of the tree remains the same as its structure during training, without any additional randomness introduced by dropout layers or changes caused by batch normalization
Th

</details> -->
