# Copyright 2023 Serket authors
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

from . import blocks
from .activation import (
    ELU,
    GELU,
    GLU,
    AdaptiveLeakyReLU,
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    CeLU,
    HardShrink,
    HardSigmoid,
    HardSwish,
    HardTanh,
    LeakyReLU,
    LogSigmoid,
    LogSoftmax,
    Mish,
    PReLU,
    ReLU,
    ReLU6,
    SeLU,
    Sigmoid,
    Snake,
    SoftPlus,
    SoftShrink,
    SoftSign,
    SquarePlus,
    Swish,
    Tanh,
    TanhShrink,
    ThresholdedReLU,
)
from .attention import MultiHeadAttention
from .blocks import UNetBlock, VGG16Block, VGG19Block
from .containers import RandomApply, Sequential
from .convolution import (
    Conv1D,
    Conv1DLocal,
    Conv1DTranspose,
    Conv2D,
    Conv2DLocal,
    Conv2DTranspose,
    Conv3D,
    Conv3DLocal,
    Conv3DTranspose,
    DepthwiseConv1D,
    DepthwiseConv2D,
    DepthwiseConv3D,
    DepthwiseFFTConv1D,
    DepthwiseFFTConv2D,
    DepthwiseFFTConv3D,
    FFTConv1D,
    FFTConv1DTranspose,
    FFTConv2D,
    FFTConv2DTranspose,
    FFTConv3D,
    FFTConv3DTranspose,
    SeparableConv1D,
    SeparableConv2D,
    SeparableConv3D,
    SeparableFFTConv1D,
    SeparableFFTConv2D,
    SeparableFFTConv3D,
)
from .dropout import (
    Dropout,
    Dropout1D,
    Dropout2D,
    Dropout3D,
    GeneralDropout,
    RandomCutout1D,
    RandomCutout2D,
)
from .image import (
    AdjustContrast2D,
    AvgBlur2D,
    FFTFilter2D,
    Filter2D,
    GaussianBlur2D,
    HistogramEqualization2D,
    HorizontalShear2D,
    Pixelate2D,
    PixelShuffle2D,
    RandomContrast2D,
    RandomHorizontalShear2D,
    RandomPerspective2D,
    RandomRotate2D,
    RandomVerticalShear2D,
    Rotate2D,
    Solarize2D,
    VerticalShear2D,
)
from .linear import FNN, MLP, Embedding, GeneralLinear, Identity, Linear, Multilinear
from .normalization import BatchNorm, GroupNorm, InstanceNorm, LayerNorm
from .pooling import (
    AdaptiveAvgPool1D,
    AdaptiveAvgPool2D,
    AdaptiveAvgPool3D,
    AdaptiveMaxPool1D,
    AdaptiveMaxPool2D,
    AdaptiveMaxPool3D,
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    GlobalAvgPool1D,
    GlobalAvgPool2D,
    GlobalAvgPool3D,
    GlobalMaxPool1D,
    GlobalMaxPool2D,
    GlobalMaxPool3D,
    LPPool1D,
    LPPool2D,
    LPPool3D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
)
from .recurrent import (
    ConvGRU1DCell,
    ConvGRU2DCell,
    ConvGRU3DCell,
    ConvLSTM1DCell,
    ConvLSTM2DCell,
    ConvLSTM3DCell,
    DenseCell,
    FFTConvGRU1DCell,
    FFTConvGRU2DCell,
    FFTConvGRU3DCell,
    FFTConvLSTM1DCell,
    FFTConvLSTM2DCell,
    FFTConvLSTM3DCell,
    GRUCell,
    LSTMCell,
    ScanRNN,
    SimpleRNNCell,
)
from .reshape import (
    Crop1D,
    Crop2D,
    Crop3D,
    Flatten,
    HorizontalFlip2D,
    Pad1D,
    Pad2D,
    Pad3D,
    RandomCrop1D,
    RandomCrop2D,
    RandomCrop3D,
    RandomZoom1D,
    RandomZoom2D,
    RandomZoom3D,
    Resize1D,
    Resize2D,
    Resize3D,
    Unflatten,
    Upsample1D,
    Upsample2D,
    Upsample3D,
    VerticalFlip2D,
)

__all__ = (
    # activation
    "ELU",
    "GELU",
    "GLU",
    "AdaptiveLeakyReLU",
    "AdaptiveReLU",
    "AdaptiveSigmoid",
    "AdaptiveTanh",
    "CeLU",
    "HardShrink",
    "HardSigmoid",
    "HardSwish",
    "HardTanh",
    "LeakyReLU",
    "LogSigmoid",
    "LogSoftmax",
    "Mish",
    "PReLU",
    "ReLU",
    "ReLU6",
    "SeLU",
    "Sigmoid",
    "Snake",
    "SoftPlus",
    "SoftShrink",
    "SoftSign",
    "SquarePlus",
    "Swish",
    "Tanh",
    "TanhShrink",
    "ThresholdedReLU",
    # attention
    "MultiHeadAttention",
    # blocks
    "UNetBlock",
    "VGG16Block",
    "VGG19Block",
    # container
    "RandomApply",
    "Sequential",
    # convolution
    "Conv1D",
    "Conv1DLocal",
    "Conv1DTranspose",
    "Conv2D",
    "Conv2DLocal",
    "Conv2DTranspose",
    "Conv3D",
    "Conv3DLocal",
    "Conv3DTranspose",
    "DepthwiseConv1D",
    "DepthwiseConv2D",
    "DepthwiseConv3D",
    "DepthwiseFFTConv1D",
    "DepthwiseFFTConv2D",
    "DepthwiseFFTConv3D",
    "FFTConv1D",
    "FFTConv1DTranspose",
    "FFTConv2D",
    "FFTConv2DTranspose",
    "FFTConv3D",
    "FFTConv3DTranspose",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConv3D",
    "SeparableFFTConv1D",
    "SeparableFFTConv2D",
    "SeparableFFTConv3D",
    # dropout
    "Dropout",
    "Dropout1D",
    "Dropout2D",
    "Dropout3D",
    "GeneralDropout",
    "RandomCutout1D",
    "RandomCutout2D",
    # linear
    "FNN",
    "MLP",
    "Embedding",
    "GeneralLinear",
    "Identity",
    "Linear",
    "Multilinear",
    # norms
    "BatchNorm",
    "GroupNorm",
    "InstanceNorm",
    "LayerNorm",
    # image
    "AdjustContrast2D",
    "AvgBlur2D",
    "FFTFilter2D",
    "Filter2D",
    "GaussianBlur2D",
    "HistogramEqualization2D",
    "HorizontalShear2D",
    "Pixelate2D",
    "PixelShuffle2D",
    "RandomContrast2D",
    "RandomHorizontalShear2D",
    "RandomPerspective2D",
    "RandomRotate2D",
    "RandomVerticalShear2D",
    "Rotate2D",
    "Solarize2D",
    "VerticalShear2D",
    # pooling
    "AdaptiveAvgPool1D",
    "AdaptiveAvgPool2D",
    "AdaptiveAvgPool3D",
    "AdaptiveMaxPool1D",
    "AdaptiveMaxPool2D",
    "AdaptiveMaxPool3D",
    "AvgPool1D",
    "AvgPool2D",
    "AvgPool3D",
    "GlobalAvgPool1D",
    "GlobalAvgPool2D",
    "GlobalAvgPool3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    "LPPool1D",
    "LPPool2D",
    "LPPool3D",
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    # rnn
    "ConvGRU1DCell",
    "ConvGRU2DCell",
    "ConvGRU3DCell",
    "ConvLSTM1DCell",
    "ConvLSTM2DCell",
    "ConvLSTM3DCell",
    "DenseCell",
    "FFTConvGRU1DCell",
    "FFTConvGRU2DCell",
    "FFTConvGRU3DCell",
    "FFTConvLSTM1DCell",
    "FFTConvLSTM2DCell",
    "FFTConvLSTM3DCell",
    "GRUCell",
    "LSTMCell",
    "ScanRNN",
    "SimpleRNNCell",
    # reshape
    "Crop1D",
    "Crop2D",
    "Crop3D",
    "Flatten",
    "HorizontalFlip2D",
    "Pad1D",
    "Pad2D",
    "Pad3D",
    "RandomCrop1D",
    "RandomCrop2D",
    "RandomCrop3D",
    "RandomZoom1D",
    "RandomZoom2D",
    "RandomZoom3D",
    "Resize1D",
    "Resize2D",
    "Resize3D",
    "Unflatten",
    "Upsample1D",
    "Upsample2D",
    "Upsample3D",
    "VerticalFlip2D",
    # block
    "blocks",
)
