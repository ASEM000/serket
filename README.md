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


#### Imports

```python
import os
os.environ["KERAS_BACKEND"] = "jax"
from keras_core.datasets import mnist  # for mnist only
import jax
import jax.numpy as jnp
import functools as ft
import optax  # for gradient optimization
import serket as sk
import time
import matplotlib.pyplot as plt  # for plotting the predictions

EPOCHS = 1
LR = 1e-3
BATCH_SIZE = 128
```

#### Data preparation

```python
(x_train, y_train), _ = mnist.load_data()

x_train = x_train.reshape(-1, 1, 28, 28).astype("float32") / 255.0
x_train = jnp.array_split(x_train, x_train.shape[0] // BATCH_SIZE)
y_train = jnp.array_split(y_train, y_train.shape[0] // BATCH_SIZE)

```

#### Model creation

_**Style 1**_
```python
k1, k2, k3 = jax.random.split(jax.random.PRNGKey(0), 3)

class ConvNet(sk.TreeClass):
    conv1: sk.nn.Conv2D = sk.nn.Conv2D(1, 32, 3, key=k1, padding="valid")
    pool1: sk.nn.MaxPool2D = sk.nn.MaxPool2D(2, 2)
    conv2: sk.nn.Conv2D = sk.nn.Conv2D(32, 64, 3, key=k2, padding="valid")
    pool2: sk.nn.MaxPool2D = sk.nn.MaxPool2D(2, 2)
    linear: sk.nn.Linear = sk.nn.Linear(1600, 10, key=k3)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self.pool1(jax.nn.relu(self.conv1(x)))
        x = self.pool2(jax.nn.relu(self.conv2(x)))
        x = self.linear(jnp.ravel(x))
        return x

nn = ConvNet()
```

_**Style 2**_
```python
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
```

#### Training functions

```python
# 1) mask the non-jaxtype parameters
nn = sk.tree_mask(nn)

# 2) initialize the optimizer state
optim = optax.adam(LR)
optim_state = optim.init(nn)

@jax.vmap
def softmax_cross_entropy(logits, onehot):
    assert onehot.shape == logits.shape == (10,)
    return -jnp.sum(jax.nn.log_softmax(logits) * onehot)

@ft.partial(jax.grad, has_aux=True)
def loss_func(nn, x, y):
    # pass non-jaxtype over jax transformation
    # using `tree_mask`/`tree_unmask` scheme
    # 3) unmask the non-jaxtype parameters to be used in the computation
    nn = sk.tree_unmask(nn)

    # 4) vectorize the computation over the batch dimension
    # and get the logits
    logits = jax.vmap(nn)(x)
    onehot = jax.nn.one_hot(y, 10)

    # 5) use the appropriate loss function
    loss = jnp.mean(softmax_cross_entropy(logits, onehot))
    return loss, (loss, logits)


@jax.vmap
def accuracy_func(logits, y):
    assert logits.shape == (10,)
    return jnp.argmax(logits) == y


@jax.jit
def train_step(nn, optim_state, x, y):
    grads, (loss, logits) = loss_func(nn, x, y)
    updates, optim_state = optim.update(grads, optim_state)
    nn = optax.apply_updates(nn, updates)
    return nn, optim_state, (loss, logits)

```

#### Train and plot results

```python

for i in range(1, EPOCHS + 1):
    t0 = time.time()
    for j, (xb, yb) in enumerate(zip(x_train, y_train)):
        nn, optim_state, (loss, logits) = train_step(nn, optim_state, xb, yb)
        accuracy = jnp.mean(accuracy_func(logits, yb))
        print(
            f"Epoch: {i:003d}/{EPOCHS:003d}\t"
            f"Batch: {j:003d}/{len(x_train):003d}\t"
            f"Batch loss: {loss:3e}\t"
            f"Batch accuracy: {accuracy:3f}\t"
            f"Time: {time.time() - t0:.3f}",
            end="\r",
        )
        
# Epoch: 001/001	Batch: 467/468	Batch loss: 2.040178e-01	Batch accuracy: 0.984375	Time: 19.284

# 6) un-mask the trained network
nn = sk.tree_unmask(nn)

# create 2x5 grid of images
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
idxs = jax.random.randint(k1, shape=(10,), minval=0, maxval=x_train[0].shape[0])

for i, idx in zip(axes.flatten(), idxs):
    # get the prediction
    pred = nn(x_train[0][idx])
    # plot the image
    i.imshow(x_train[0][idx].reshape(28, 28), cmap="gray")
    # set the title to be the prediction
    i.set_title(jnp.argmax(pred))
    i.set_xticks([])
    i.set_yticks([])
```