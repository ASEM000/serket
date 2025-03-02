<div align="center">
<img width="150px" src=https://github.com/ASEM000/serket/assets/48389287/1ea9efd6-d848-48dc-9342-4198a9d9a90c></div>

<h2 align="center">The ✨Magical✨ JAX ML Library.</h2>
<h5 align = "center"> *Serket is the goddess of magic in Egyptian mythology

[**Installation**](#Installation)
|[**Description**](#Description)
|[**Documentation**](#Documentation)
|[**Quick Example**](#QuickExample)

![Tests](https://github.com/ASEM000/serket/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.10%203.11%203.12%203.13-blue)
![codestyle](https://img.shields.io/badge/codestyle-black-black)
[![codecov](https://codecov.io/gh/ASEM000/serket/branch/main/graph/badge.svg?token=C6NXOK9EVS)](https://codecov.io/gh/ASEM000/serket)
[![Documentation Status](https://readthedocs.org/projects/serket/badge/?version=latest)](https://serket.readthedocs.io/?badge=latest)
[![DOI](https://zenodo.org/badge/526985786.svg)](https://zenodo.org/badge/latestdoi/526985786)
[![CodeFactor](https://www.codefactor.io/repository/github/asem000/serket/badge)](https://www.codefactor.io/repository/github/asem000/serket)

</h5>

## 🛠️ Installation<a id="Installation"></a>

**Install development version**

```python
pip install git+https://github.com/ASEM000/serket
```

## 📖 Description and motivation<a id="Description"></a>

- `serket` aims to be the most intuitive and easy-to-use machine learning library in `jax`.
- `serket` is fully transparent to `jax` transformation (e.g. `vmap`,`grad`,`jit`,...).

## 📙 Documentation <a id="Documentation"></a>
- [Full documentation](https://serket.readthedocs.io/)
- [Train MNIST, UNet, ConvLSTM, PINN](https://serket.readthedocs.io/training_guides.html)
- [Model surgery, Parallelism, Mixed precision](https://serket.readthedocs.io/core_guides.html)
- [Optimizers, Augmentation composition](https://serket.readthedocs.io/other_guides.html)
- [Interoperability with keras, tensorflow](https://serket.readthedocs.io/interoperability.html)


## 🏃 Quick example<a id="QuickExample"></a>

```python
import jax, jax.numpy as jnp
import serket as sk

x_train, y_train = ..., ...
k1, k2 = jax.random.split(jax.random.key(0))

net = sk.tree_mask(sk.Sequential(
    jnp.ravel,
    sk.nn.Linear(28 * 28, 64, key=k1),
    jax.nn.relu,
    sk.nn.Linear(64, 10, key=k2),
))

@ft.partial(jax.grad, has_aux=True)
def loss_func(net, x, y):
    logits = jax.vmap(sk.tree_unmask(net))(x)
    onehot = jax.nn.one_hot(y, 10)
    loss = jnp.mean(softmax_cross_entropy(logits, onehot))
    return loss, (loss, logits)

@jax.jit
def train_step(net, x, y):
    grads, (loss, logits) = loss_func(net, x, y)
    net = jax.tree_map(lambda p, g: p - g * 1e-3, net, grads)
    return net, (loss, logits)

for j, (xb, yb) in enumerate(zip(x_train, y_train)):
    net, (loss, logits) = train_step(net, xb, yb)
    accuracy = accuracy_func(logits, y_train)

net = sk.tree_unmask(net)
```

<details> <summary> 📚 Layers catalog </summary>

#### 🔗 Common API

| Group      | Layers                           |
| ---------- | -------------------------------- |
| Containers | - `Sequential`, `Random{Choice}` |

#### 🧠 Neural network package: `serket.nn`

| Group             | Layers                                                                                                                                                                                                                                                                                                                    |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Attention         | - `MultiHeadAttention`                                                                                                                                                                                                                                                                                                    |
| Convolution       | - `{FFT,_}Conv{1D,2D,3D}` <br> - `{FFT,_}Conv{1D,2D,3D}Transpose` <br> - `Depthwise{FFT,_}Conv{1D,2D,3D}` <br> - `Separable{FFT,_}Conv{1D,2D,3D}` <br> - `Conv{1D,2D,3D}Local` <br> - `SpectralConv{1D,2D,3D}`                                                                                                            |
| Dropout           | - `Dropout`<br> - `Dropout{1D,2D,3D}` <br> - `RandomCutout{1D,2D,3D}`                                                                                                                                                                                                                                                     |
| Linear            | - `Linear`, `MLP`, `Identity`                                                                                                                                                                                                                                                                                   |                                                                                                                                                                                                                                                            |
| Normalization     | - `{Layer,Instance,Group,Batch}Norm`                                                                                                                                                                                                                                                                                      |
| Pooling           | - `{Avg,Max,LP}Pool{1D,2D,3D}` <br> - `Global{Avg,Max}Pool{1D,2D,3D}` <br> - `Adaptive{Avg,Max}Pool{1D,2D,3D}`                                                                                                                                                                                                            |
| Reshaping         | - `Upsample{1D,2D,3D}` <br> - `{Random,Center}Crop{1D,2D,3D}` `                                                                                                                                                                                                                                                           |
| Recurrent cells   | - `{SimpleRNN,LSTM,GRU,Dense}Cell` <br> - `{Conv,FFTConv}{LSTM,GRU}{1D,2D,3D}Cell`                                                                                                                                                                                                                                        |
| Activations       | - `Adaptive{LeakyReLU,ReLU,Sigmoid,Tanh}`,<br> - `CeLU`,`ELU`,`GELU`,`GLU`<br>- `Hard{SILU,Shrink,Sigmoid,Swish,Tanh}`, <br> - `Soft{Plus,Sign,Shrink}` <br> - `LeakyReLU`,`LogSigmoid`,`LogSoftmax`,`Mish`,`PReLU`,<br> - `ReLU`,`ReLU6`,`SeLU`,`Sigmoid` <br> - `Swish`,`Tanh`,`TanhShrink`, `ThresholdedReLU`, `Snake` |

#### 🖼️ Image package: `serket.image`

| Group     | Layers                                                                                                                                                                                                           |
| --------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Filter    | - `{FFT,_}{Avg,Box,Gaussian,Motion}Blur2D` <br> - `{JointBilateral,Bilateral,Median}Blur2D` <br> - `{FFT,_}{UnsharpMask}2D` <br> - `{FFT,_}{Sobel,Laplacian}2D` <br> - `{FFT,_}BlurPool2D`                       |
| Augment   | - `Adjust{Sigmoid,Log}2D` <br> - `{Adjust,Random}{Brightness,Contrast,Hue,Saturation}2D`, <br> - `RandomJigSaw2D`,`PixelShuffle2D`, <br> - `Pixelate2D`,`Posterize2D`,`Solarize2D` <br> - `FourierDomainAdapt2D` |
| Geometric | - `{Random,_}{Horizontal,Vertical}{Translate,Flip,Shear}2D` <br> - `{Random,_}{Rotate}2D` <br> - `RandomPerspective2D` <br> - `{FFT,_}ElasticTransform2D`                     |
| Color     | - `RGBToGrayscale2D` , `GrayscaleToRGB2D` <br> - `RGBToHSV2D`, `HSVToRGB2D`                                                                                                                                      |

</details>
