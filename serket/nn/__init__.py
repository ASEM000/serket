from . import blocks
from .activation import (
    ELU,
    GELU,
    GLU,
    SILU,
    AdaptiveLeakyReLU,
    AdaptiveReLU,
    AdaptiveSigmoid,
    AdaptiveTanh,
    CeLU,
    HardShrink,
    HardSigmoid,
    HardSILU,
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
    SoftPlus,
    SoftShrink,
    SoftSign,
    Swish,
    Tanh,
    TanhShrink,
    ThresholdedReLU,
)
from .blocks import UNetBlock, VGG16Block, VGG19Block
from .blur import AvgBlur2D, GaussianBlur2D
from .containers import Lambda, Sequential
from .contrast import AdjustContrast2D, RandomContrast2D
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
    SeparableConv1D,
    SeparableConv2D,
    SeparableConv3D,
)

from .crop import Crop1D, Crop2D, Crop3D, RandomCrop1D, RandomCrop2D, RandomCrop3D
from .cutout import RandomCutout1D, RandomCutout2D
from .dropout import Dropout, Dropout1D, Dropout2D, Dropout3D
from .flatten import Flatten, Unflatten
from .flip import FlipLeftRight2D, FlipUpDown2D
from .fully_connected import FNN, PFNN
from .laplace import Laplace2D
from .linear import Bilinear, Identity, Linear
from .normalization import GroupNorm, InstanceNorm, LayerNorm
from .padding import Padding1D, Padding2D, Padding3D
from .pooling import (
    AvgPool1D,
    AvgPool2D,
    AvgPool3D,
    GlobalAvgPool1D,
    GlobalAvgPool2D,
    GlobalAvgPool3D,
    GlobalMaxPool1D,
    GlobalMaxPool2D,
    GlobalMaxPool3D,
    MaxPool1D,
    MaxPool2D,
    MaxPool3D,
)
from .preprocessing import HistogramEqualization2D
from .random_transform import RandomApply, RandomZoom2D
from .resize import (
    Repeat1D,
    Repeat2D,
    Repeat3D,
    Resize1D,
    Resize2D,
    Resize3D,
    Upsampling1D,
    Upsampling2D,
    Upsampling3D,
)

__all__ = (
    "blocks",
    # Fully connected
    "FNN",
    "PFNN",
    # Linear
    "Linear",
    "Bilinear",
    "Identity",
    # Dropout
    "Dropout",
    "Dropout1D",
    "Dropout2D",
    "Dropout3D",
    # containers
    "Sequential",
    "Lambda",
    # Pooling
    "MaxPool1D",
    "MaxPool2D",
    "MaxPool3D",
    "AvgPool1D",
    "AvgPool2D",
    "AvgPool3D",
    "GlobalAvgPool1D",
    "GlobalAvgPool2D",
    "GlobalAvgPool3D",
    "GlobalMaxPool1D",
    "GlobalMaxPool2D",
    "GlobalMaxPool3D",
    # Convolution
    "Conv1D",
    "Conv2D",
    "Conv3D",
    "Conv1DTranspose",
    "Conv2DTranspose",
    "Conv3DTranspose",
    "DepthwiseConv1D",
    "DepthwiseConv2D",
    "DepthwiseConv3D",
    "SeparableConv1D",
    "SeparableConv2D",
    "SeparableConv3D",
    "Conv1DLocal",
    "Conv2DLocal",
    "Conv3DLocal",
    # Flattening
    "Flatten",
    "Unflatten",
    "Repeat1D",
    "Repeat2D",
    "Repeat3D",
    # Normalization
    "LayerNorm",
    "InstanceNorm",
    "GroupNorm",
    # Blur
    "AvgBlur2D",
    "GaussianBlur2D",
    # Resize
    "Laplace2D",
    "FlipLeftRight2D",
    "FlipUpDown2D",
    "Resize1D",
    "Resize2D",
    "Resize3D",
    "Upsampling1D",
    "Upsampling2D",
    "Upsampling3D",
    "Padding1D",
    "Padding2D",
    "Padding3D",
    # blocks
    "VGG16Block",
    "VGG19Block",
    "UNetBlock",
    # Crop
    "Crop1D",
    "Crop2D",
    "Crop3D",
    # Random Transform
    "RandomCrop1D",
    "RandomCrop2D",
    "RandomCrop3D",
    "RandomCutout1D",
    "RandomCutout2D",
    "RandomZoom2D",
    "RandomApply",
    "HistogramEqualization2D",
    # Activations
    "AdaptiveLeakyReLU",
    "AdaptiveReLU",
    "AdaptiveSigmoid",
    "AdaptiveTanh",
    "CeLU",
    "ELU",
    "GELU",
    "GLU",
    "HardSILU",
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
    "SILU",
    "SeLU",
    "Sigmoid",
    "SoftPlus",
    "SoftShrink",
    "SoftSign",
    "Swish",
    "Tanh",
    "TanhShrink",
    "ThresholdedReLU",
    # Contrast
    "AdjustContrast2D",
    "RandomContrast2D",
)
