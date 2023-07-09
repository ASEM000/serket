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
![codestyle](https://img.shields.io/badge/codestyle-black-lightgrey)
[![Downloads](https://pepy.tech/badge/serket)](https://pepy.tech/project/serket)
[![codecov](https://codecov.io/gh/ASEM000/serket/branch/main/graph/badge.svg?token=C6NXOK9EVS)](https://codecov.io/gh/ASEM000/serket)
[![DOI](https://zenodo.org/badge/526985786.svg)](https://zenodo.org/badge/latestdoi/526985786)
![PyPI](https://img.shields.io/pypi/v/serket)

</h5>

## 🛠️ Installation<a id="Installation"></a>

**Install development version**

```python
pip install git+https://github.com/ASEM000/serket
```

## 📖 Description and motivation<a id="Description"></a>

- `serket` aims to be the most intuitive and easy-to-use physics-based Neural network library in JAX.
- `serket` is fully transparent to `jax` transformation (e.g. `vmap`,`grad`,`jit`,...)
- `serket` current aim to facilitate the integration of numerical methods in a NN setting (see examples for more)

<div align="center">

### 🧠 Neural network package: `serket.nn` 🧠

| Group                           | Layers                                                                                                                                                                                                                                                                                                                                                 |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Linear                          | - `Linear`, `Bilinear`, `Multilinear`, `GeneralLinear`, `Identity`, `Embedding`                                                                                                                                                                                                                                                                        |
| Densely connected               | - `FNN` (Fully connected network),                                                                                                                                                                                                                                                                                                                     |
| Convolution                     | - `{Conv,FFTConv}{1D,2D,3D}` <br> - `{Conv,FFTConv}{1D,2D,3D}Transpose` <br> - `{Depthwise,Separable}{Conv,FFTConv}{1D,2D,3D}` <br> - `Conv{1D,2D,3D}Local`                                                                                                                                                                                            |
| Containers                      | - `Sequential`, `Lambda`                                                                                                                                                                                                                                                                                                                               |
| Pooling <br> (`kernex` backend) | - `{Avg,Max,LP}Pool{1D,2D,3D}` <br> - `Global{Avg,Max}Pool{1D,2D,3D}` <br> - `Adaptive{Avg,Max}Pool{1D,2D,3D}`                                                                                                                                                                                                                                         |
| Reshaping                       | - `Flatten`, `Unflatten`, <br> - `FlipLeftRight2D`, `FlipUpDown2D` <br> - `Resize{1D,2D,3D}` <br> - `Upsample{1D,2D,3D}` <br> - `Pad{1D,2D,3D}`                                                                                                                                                                                                        |
| Crop                            | - `Crop{1D,2D}`                                                                                                                                                                                                                                                                                                                                        |
| Normalization                   | - `{Layer,Instance,Group}Norm`                                                                                                                                                                                                                                                                                                                         |
| Blurring                        | - `{Avg,Gaussian}Blur2D`                                                                                                                                                                                                                                                                                                                               |
| Dropout                         | - `Dropout`<br> - `Dropout{1D,2D,3D}`                                                                                                                                                                                                                                                                                                                  |
| Random transforms               | - `RandomCrop{1D,2D}` <br> - `RandomApply`, <br> - `RandomCutout{1D,2D}` <br> - `RandomZoom2D`, <br> - `RandomContrast2D`                                                                                                                                                                                                                              |
| Misc                            | - `HistogramEqualization2D`, `AdjustContrast2D`, `Filter2D`, `PixelShuffle2D`                                                                                                                                                                                                                                                                          |
| Activations                     | - `Adaptive{LeakyReLU,ReLU,Sigmoid,Tanh}`,<br> - `CeLU`,`ELU`,`GELU`,`GLU`<br>- `Hard{SILU,Shrink,Sigmoid,Swish,Tanh}`, <br> - `Soft{Plus,Sign,Shrink}` <br> - `LeakyReLU`,`LogSigmoid`,`LogSoftmax`,`Mish`,`PReLU`,<br> - `ReLU`,`ReLU6`,`SILU`,`SeLU`,`Sigmoid` <br> - `Swish`,`Tanh`,`TanhShrink`, `ThresholdedReLU`, `Snake`, `Stan`, `SquarePlus` |
| Recurrent cells                 | - `{SimpleRNN,LSTM,GRU}Cell` <br> - `Conv{LSTM,GRU}{1D,2D,3D}Cell`                                                                                                                                                                                                                                                                                     |
| Blocks                          | - `VGG{16,19}Block`, `UNetBlock`                                                                                                                                                                                                                                                                                                                       |

</div>
