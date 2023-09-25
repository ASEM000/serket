<div align="center">
<img width="250px" src="https://github.com/ASEM000/serket/assets/48389287/1733c849-c44f-4810-b74a-8a10263e8cc5"></div>

<h2 align="center">The ✨Magical✨ JAX ML Library.</h2>
<h5 align = "center"> *Serket is the goddess of magic in Egyptian mythology

[**Installation**](#Installation)
|[**Description**](#Description)
|[**Quick Example**](#QuickExample)

![Tests](https://github.com/ASEM000/serket/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.8%203.9%203.10%203.11-blue)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![Downloads](https://static.pepy.tech/badge/serket)](https://pepy.tech/project/serket)
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

See [🧠 `serket` mental model](https://serket.readthedocs.io/en/latest/notebooks/mental_model.html) and for examples, see [Training 🚆 MNIST](https://serket.readthedocs.io/en/latest/notebooks/train_mnist.html), or [Training 🚆 Bidirectional-LSTM](https://serket.readthedocs.io/en/latest/notebooks/train_bilstm.html)

```python
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
```

<details>

<summary>  🧱 Layers catalog </summary>

### 🧠 Neural network package: `serket.nn`

| Group             | Layers                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Attention         | - `MultiHeadAttention`                                                                                                                                                                                                                                                                                                    |
| Containers        | - `Sequential`, `RandomApply`, `RandomChoice`                                                                                                                                                                                                                                                                             |
| Convolution       | - `{FFT,_}Conv{1D,2D,3D}` <br> - `{FFT,_}Conv{1D,2D,3D}Transpose` <br> - `Depthwise{FFT,_}Conv{1D,2D,3D}` <br> - `Separable{FFT,_}Conv{1D,2D,3D}` <br> - `Conv{1D,2D,3D}Local`                                                                                                                                            |
| Dropout           | - `Dropout`<br> - `Dropout{1D,2D,3D}` <br> - `RandomCutout{1D,2D,3D}`                                                                                                                                                                                                                                                        |
| Linear            | - `Linear`, `Multilinear`, `GeneralLinear`, `Identity`                                                                                                                                                                                                                                                                    |
| Densely connected | - `FNN` , <br> - `MLP` _compile time_ optimized                                                                                                                                                                                                                                                                           |
| Normalization     | - `{Layer,Instance,Group,Batch}Norm`                                                                                                                                                                                                                                                                                      |
| Pooling           | - `{Avg,Max,LP}Pool{1D,2D,3D}` <br> - `Global{Avg,Max}Pool{1D,2D,3D}` <br> - `Adaptive{Avg,Max}Pool{1D,2D,3D}`                                                                                                                                                                                                            |
| Reshaping         | - `Flatten`, `Unflatten`, <br> - `Resize{1D,2D,3D}` <br> - `Upsample{1D,2D,3D}` <br> - `Pad{1D,2D,3D}` <br> - `{Random,Center,_}Crop{1D,2D,3D}` <br> - `RandomZoom{1D,2D,3D}`                                                                                                                                             |
| Recurrent cells   | - `{SimpleRNN,LSTM,GRU,Dense}Cell` <br> - `{Conv,FFTConv}{LSTM,GRU}{1D,2D,3D}Cell`                                                                                                                                                                                                                                        |
| Activations       | - `Adaptive{LeakyReLU,ReLU,Sigmoid,Tanh}`,<br> - `CeLU`,`ELU`,`GELU`,`GLU`<br>- `Hard{SILU,Shrink,Sigmoid,Swish,Tanh}`, <br> - `Soft{Plus,Sign,Shrink}` <br> - `LeakyReLU`,`LogSigmoid`,`LogSoftmax`,`Mish`,`PReLU`,<br> - `ReLU`,`ReLU6`,`SeLU`,`Sigmoid` <br> - `Swish`,`Tanh`,`TanhShrink`, `ThresholdedReLU`, `Snake` |

### 🖼️ Image package: `serket.image`

| Group     | Layers                                                                                                                                 |
| --------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Filter    | - `{FFT,_}{Avg,Box,Gaussian,Motion}Blur2D` <br> - `{FFT,_}{UnsharpMask}2D` <br> - `{FFT,_}Laplacian2D` <br> - `MedianBlur2D`           |
| Augment   | - `{Adjust,Random}{Brightness,Contrast}2D`, <br> - `JigSaw2D`,`PixelShuffle2D`, <br> - `Pixelate2D`, <br> - `Posterize2D`,`Solarize2D` |
| Geometric | - `{Random,_}{Horizontal,Vertical}{Translate,Flip,Shear}2D` <br> - `{Random,_}{Rotate}2D` <br> - `RandomPerspective2D`                 |

### 🌈 Cluster package: `serket.cluster`

| Group      | Layers     |
| ---------- | ---------- |
| Clustering | - `KMeans` |

</details>
