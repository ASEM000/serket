# see : https://pytorch.org/hub/mateuszbuda_brain-segmentation-pytorch_unet/
# current implementation is based on the above link


from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

import serket as sk


class ResizeAndCat(pytc.TreeClass):
    def __call__(self, x1: jax.Array, x2: jax.Array) -> jax.Array:
        """resize a tensor to the same size as another tensor and concatenate x2 to x1 along the channel axis"""
        x1 = jax.image.resize(x1, shape=x2.shape, method="nearest")
        x1 = jnp.concatenate([x2, x1], axis=0)
        return x1


class DoubleConvBlock(pytc.TreeClass):
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

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        return x


class UpscaleBlock(pytc.TreeClass):
    def __init__(self, in_features: int, out_features: int):
        self.conv = sk.nn.Conv2DTranspose(
            in_features=in_features, out_features=out_features, kernel_size=2, strides=2
        )

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
        # x = self.upscale(x)
        x = self.conv(x)
        return x


class UNetBlock(pytc.TreeClass):
    in_features: int = pytc.field(callbacks=[pytc.freeze])
    out_features: int = pytc.field(callbacks=[pytc.freeze])
    blocks: int = pytc.field(callbacks=[pytc.freeze])
    init_features: int = pytc.field(callbacks=[pytc.freeze])

    def __init__(
        self,
        in_features: int,
        out_features: int,
        blocks: int = 4,
        init_features: int = 64,
    ):
        """Vanilla UNet

        Args:
            in_features : number of input channels. This is the number of channels in the input image.
            out_features : number of output channels. This is the number of classes
            blocks : number of blocks in the UNet architecture . Default is 4
            init_features : number of features in the first block. Default is 64


        Summary:
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

        Two block example:
            >>> print(sk.nn.UNetBlock(3,1,blocks=2, init_features=32).summary(show_config=False))
            ┌──────────┬────────────────────────────┬──────────┬───────────────┐
            │Name      │Type                        │Param #   │Size           │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d0_1/conv1│DoubleConvBlock/Conv2D      │864(0)    │3.38KB(0.00B)  │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d0_1/conv2│DoubleConvBlock/Conv2D      │9,216(0)  │36.00KB(0.00B) │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d0_2      │MaxPool2D                   │0(0)      │0.00B(0.00B)   │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d1_1/conv1│DoubleConvBlock/Conv2D      │18,432(0) │72.00KB(0.00B) │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d1_1/conv2│DoubleConvBlock/Conv2D      │36,864(0) │144.00KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │d1_2      │MaxPool2D                   │0(0)      │0.00B(0.00B)   │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │b0_1/conv1│DoubleConvBlock/Conv2D      │73,728(0) │288.00KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │b0_1/conv2│DoubleConvBlock/Conv2D      │147,456(0)│576.00KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u1_1/conv │UpscaleBlock/Conv2DTranspose│32,832(0) │128.25KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u1_2      │ResizeAndCat                │0(0)      │0.00B(0.00B)   │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u1_3/conv1│DoubleConvBlock/Conv2D      │73,728(0) │288.00KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u1_3/conv2│DoubleConvBlock/Conv2D      │36,864(0) │144.00KB(0.00B)│
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u0_1/conv │UpscaleBlock/Conv2DTranspose│8,224(0)  │32.12KB(0.00B) │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u0_2      │ResizeAndCat                │0(0)      │0.00B(0.00B)   │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u0_3/conv1│DoubleConvBlock/Conv2D      │18,432(0) │72.00KB(0.00B) │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │u0_3/conv2│DoubleConvBlock/Conv2D      │9,216(0)  │36.00KB(0.00B) │
            ├──────────┼────────────────────────────┼──────────┼───────────────┤
            │f0_1      │Conv2D                      │33(0)     │132.00B(0.00B) │
            └──────────┴────────────────────────────┴──────────┴───────────────┘
            Total count :	465,889(0)
            Dynamic count :	465,889(0)
            Frozen count :	0(0)
            --------------------------------------------------------------------
            Total size :	1.78MB(0.00B)
            Dynamic size :	1.78MB(0.00B)
            Frozen size :	0.00B(0.00B)
            ====================================================================
        """

        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.init_features = init_features

        self.d0_1 = DoubleConvBlock(in_features, init_features)
        self.d0_2 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        for i in range(1, blocks):
            # conv->relu x2
            layer = DoubleConvBlock(init_features * (2 ** (i - 1)), init_features * (2**i))  # fmt: skip
            setattr(self, f"d{i}_1", layer)
            setattr(self, f"d{i}_2", sk.nn.MaxPool2D(kernel_size=2, strides=2))

        self.b0_1 = DoubleConvBlock(init_features * (2 ** (blocks - 1)), init_features * (2 ** (blocks)))  # fmt: skip

        for i in range(blocks, 0, -1):
            # upscale and conv to halves channels size and double row,col size
            layer = UpscaleBlock(init_features * (2 ** (i)), init_features * (2 ** (i - 1)))  # fmt: skip
            setattr(self, f"u{i-1}_1", layer)

            layer = ResizeAndCat()
            setattr(self, f"u{i-1}_2", layer)

            layer = DoubleConvBlock(init_features * (2 ** (i)), init_features * (2 ** (i - 1)))  # fmt: skip
            setattr(self, f"u{i-1}_3", layer)

        self.f0_1 = sk.nn.Conv2D(init_features, out_features, kernel_size=1)

    def __call__(self, x: jax.Array) -> jax.Array:
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
        result[f"u{blocks-1}_2"] = getattr(self, f"u{blocks-1}_2")(result[lhs_key], result[rhs_key])  # fmt: skip
        result[f"u{blocks-1}_3"] = getattr(self, f"u{blocks-1}_3")(result[f"u{blocks-1}_2"])  # fmt: skip

        for i in range(blocks - 1, 0, -1):
            result[f"u{i-1}_1"] = getattr(self, f"u{i-1}_1")(result[f"u{i}_3"])
            result[f"u{i-1}_2"] = getattr(self, f"u{blocks-1}_2")(result[f"u{i-1}_1"], result[f"d{i-1}_1"])  # fmt: skip
            result[f"u{i-1}_3"] = getattr(self, f"u{i-1}_3")(result[f"u{i-1}_2"])

        return self.f0_1(result["u0_3"])
