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

# see : https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
# current implementation is based on the above link


from __future__ import annotations

import jax
import jax.numpy as jnp

import serket as sk
from serket.nn.utils import positive_int_cb


class ResizeAndCat(sk.TreeClass):
    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """resize a tensor to the same size as another tensor and concatenate x2 to x1 along the channel axis"""
        x1 = jax.image.resize(x1, shape=x2.shape, method="nearest")
        x1 = jnp.concatenate([x2, x1], axis=0)
        return x1


class DoubleConvBlock(sk.TreeClass):
    def __init__(self, in_features: int, out_features: int):
        self.conv1 = sk.nn.Conv2D(
            in_features=in_features,
            out_features=out_features,
            kernel_size=3,
            padding=1,
            bias_init_func=None,
        )
        self.conv2 = sk.nn.Conv2D(
            in_features=out_features,
            out_features=out_features,
            kernel_size=3,
            padding=1,
            bias_init_func=None,
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        return x


class UpscaleBlock(sk.TreeClass):
    def __init__(self, in_features: int, out_features: int):
        self.conv = sk.nn.Conv2DTranspose(
            in_features=in_features,
            out_features=out_features,
            kernel_size=2,
            strides=2,
        )

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        return self.conv(x)


class UNetBlock(sk.TreeClass):
    """Vanilla UNet

    Args:
        in_features : number of input channels. This is the number of channels in the input image.
        out_features : number of output channels. This is the number of classes
        blocks : number of blocks in the UNet architecture . Default is 4
        init_features : number of features in the first block. Default is 64
    """

    in_features: int = sk.field(callbacks=[positive_int_cb])
    out_features: int = sk.field(callbacks=[positive_int_cb])
    blocks: int = sk.field(callbacks=[positive_int_cb], default=4)
    init_features: int = sk.field(callbacks=[positive_int_cb], default=64)

    def __post_init__(self):
        """
        Note:
            d0_1 :
                block_number = 0 , operation = (conv->relu) x2
            d0_2 :
                block_number = 0 , operation = maxpool previous output
            u0_1 :
                expansive block corresponding to block 0 in contractive path ,
                operation = doubling row,col size and halving channels size of previous layer
            u0_2 :
                expansive block corresponding to block 0 in contractive path ,
                operation = pad the previous layer from expansive path (u0_1) and concatenate with corresponding
                layer from contractive path (d0_1)
            u0_3 :
                expansive block corresponding to block 0 in contractive path ,
                operation = (conv->relu) x2 of previous layer (u0_2)
            b0_1 :
                bottleneck layer
            f0_1 :
                final output layer

        """
        self.d0_1 = DoubleConvBlock(self.in_features, self.init_features)
        self.d0_2 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        for i in range(1, self.blocks):
            in_dim = self.init_features * (2 ** (i - 1))
            out_dim = self.init_features * (2**i)

            layer = DoubleConvBlock(in_dim, out_dim)
            setattr(self, f"d{i}_1", layer)
            setattr(self, f"d{i}_2", sk.nn.MaxPool2D(kernel_size=2, strides=2))

        self.b0_1 = DoubleConvBlock(
            self.init_features * (2 ** (self.blocks - 1)),
            self.init_features * (2 ** (self.blocks)),
        )

        for i in range(self.blocks, 0, -1):
            # upscale and conv to halves channels size and double row,col size
            in_dim = self.init_features * (2 ** (i - 1))
            out_dim = self.init_features * (2**i)

            layer = UpscaleBlock(out_dim, in_dim)
            setattr(self, f"u{i-1}_1", layer)

            layer = ResizeAndCat()
            setattr(self, f"u{i-1}_2", layer)

            layer = DoubleConvBlock(out_dim, in_dim)
            setattr(self, f"u{i-1}_3", layer)

        self.f0_1 = sk.nn.Conv2D(self.init_features, self.out_features, kernel_size=1)

    def __call__(self, x: jax.Array, **k) -> jax.Array:
        # TODO: fix to not record intermediate results
        result = dict()
        blocks = self.blocks

        # contractive path
        result["d0_1"] = self.d0_1(x)
        result["d0_2"] = self.d0_2(result["d0_1"])

        for i in range(1, blocks):
            result[f"d{i}_1"] = getattr(self, f"d{i}_1")(result[f"d{i-1}_2"])
            result[f"d{i}_2"] = getattr(self, f"d{i}_2")(result[f"d{i}_1"])

        result["b0_1"] = self.b0_1(result[f"d{blocks-1}_2"])

        result[f"u{blocks-1}_1"] = getattr(self, f"u{blocks-1}_1")(result["b0_1"])
        lhs_key, rhs_key = f"u{blocks-1}_1", f"d{blocks-1}_1"
        result[f"u{blocks-1}_2"] = getattr(self, f"u{blocks-1}_2")(
            result[lhs_key], result[rhs_key]
        )
        result[f"u{blocks-1}_3"] = getattr(self, f"u{blocks-1}_3")(
            result[f"u{blocks-1}_2"]
        )

        for i in range(blocks - 1, 0, -1):
            result[f"u{i-1}_1"] = getattr(self, f"u{i-1}_1")(result[f"u{i}_3"])
            result[f"u{i-1}_2"] = getattr(self, f"u{blocks-1}_2")(
                result[f"u{i-1}_1"], result[f"d{i-1}_1"]
            )
            result[f"u{i-1}_3"] = getattr(self, f"u{i-1}_3")(result[f"u{i-1}_2"])

        return self.f0_1(result["u0_3"])
