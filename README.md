<div align="center">
<img width="350px" src="assets/logo.svg"></div>

<h2 align="center">The ✨Magical✨ JAX Scientific ML Library.</h2>
<h5 align = "center"> *Serket is the goddess of magic in Egyptian mythology

[**Installation**](#Installation)
|[**Description**](#Description)
|[**Quick Example**](#QuickExample)
|[**Freezing/Fine tuning**](#Freezing)
|[**Filtering**](#Filtering)

![Tests](https://github.com/ASEM000/serket/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.7%203.8%203.9%203.10-red)
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

### Quick example

For full examples see [here](https://serket.readthedocs.io/en/latest/examples.html) e.g. [Training 🚆 MNIST](https://serket.readthedocs.io/en/latest/notebooks/mnist.html), or [Training 🚆 Bidirectional-LSTM](https://serket.readthedocs.io/en/latest/notebooks/bilstm.html)

```python
import jax, jax.numpy as jnp
import serket as sk
import optax

x_train = ...
y_train = ...

k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

nn = sk.nn.Sequential(
    sk.nn.Conv2D(1, 32, 3, key=k1, padding="valid"),
    jax.nn.relu,
    sk.nn.MaxPool2D(2, 2),
    sk.nn.Conv2D(32, 64, 3, key=k2, padding="valid"),
    jax.nn.relu,
    sk.nn.MaxPool2D(2, 2),
    jnp.ravel,
    sk.nn.Linear(1600, 10, key=k3),
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
```