from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

import serket as sk


def _resize_and_cat(x1: jnp.ndarray, x2: jnp.ndarray) -> jnp.ndarray:
    """resize a tensor to the same size as another tensor and concatenate x2 to x1 along the channel axis"""
    x1 = jax.image.resize(x1, shape=x2.shape, method="nearest")
    x1 = jnp.concatenate([x2, x1], axis=0)
    return x1


@pytc.treeclass
class DoubleConvBlock:
    def __init__(self, in_features: int, out_features: int):
        self.conv1 = sk.nn.Conv2D(in_features, out_features, kernel_size=3, padding=1)
        self.conv2 = sk.nn.Conv2D(out_features, out_features, kernel_size=3, padding=1)

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.conv1(x)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = jax.nn.relu(x)
        return x


@pytc.treeclass
class UpscaleBlock:
    def __init__(self, in_features: int, out_features: int):
        self.upscale = sk.nn.Upsampling2D(scale=2, method="bilinear")
        self.conv = sk.nn.Conv2D(
            in_features, out_features, kernel_size=1, padding="valid"
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.upscale(x)
        x = self.conv(x)
        return x


@pytc.treeclass
class UNetBlock:
    in_features: int = pytc.nondiff_field()
    out_features: int = pytc.nondiff_field()
    blocks: int = pytc.nondiff_field()
    init_filters: int = pytc.nondiff_field()

    def __init__(
        self,
        in_features: int,
        out_features: int,
        blocks: int = 4,
        init_filters: int = 64,
    ):
        """Vanilla UNet

        Args:
            in_features (int): number of input features
            out_features (int): number of output features
            blocks (int, optional): number of blocks in a single path. Defaults to 4.
            init_filters (int, optional): number of filters in the first block. Defaults to init_filters.


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
            >>> print(UNet(3, 1, 2).summary(compact=True))
            ┌────────────┬─────────────────────────┬──────────┬───────────────┐
            │Name        │Type                     │Param #   │Size           │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d0_1/conv1  │DoubleConvBlock/Conv2D   │1,792(0)  │7.00KB(0.00B)  │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d0_1/conv2  │DoubleConvBlock/Conv2D   │36,928(0) │144.25KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d0_2        │MaxPool2D                │0(0)      │0.00B(0.00B)   │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d1_1/conv1  │DoubleConvBlock/Conv2D   │73,856(0) │288.50KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d1_1/conv2  │DoubleConvBlock/Conv2D   │147,584(0)│576.50KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │d1_2        │MaxPool2D                │0(0)      │0.00B(0.00B)   │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │b0_1/conv1  │DoubleConvBlock/Conv2D   │295,168(0)│1.13MB(0.00B)  │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │b0_1/conv2  │DoubleConvBlock/Conv2D   │590,080(0)│2.25MB(0.00B)  │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u1_1/upscale│UpscaleBlock/Upsampling2D│0(0)      │0.00B(0.00B)   │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u1_1/conv   │UpscaleBlock/Conv2D      │32,896(0) │128.50KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u1_3/conv1  │DoubleConvBlock/Conv2D   │295,040(0)│1.13MB(0.00B)  │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u1_3/conv2  │DoubleConvBlock/Conv2D   │147,584(0)│576.50KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u0_1/upscale│UpscaleBlock/Upsampling2D│0(0)      │0.00B(0.00B)   │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u0_1/conv   │UpscaleBlock/Conv2D      │8,256(0)  │32.25KB(0.00B) │
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u0_3/conv1  │DoubleConvBlock/Conv2D   │73,792(0) │288.25KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │u0_3/conv2  │DoubleConvBlock/Conv2D   │36,928(0) │144.25KB(0.00B)│
            ├────────────┼─────────────────────────┼──────────┼───────────────┤
            │f0_1        │Conv2D                   │65(0)     │260.00B(0.00B) │
            └────────────┴─────────────────────────┴──────────┴───────────────┘
            Total count :	1,739,969(0)
            Dynamic count :	1,739,969(0)
            Frozen count :	0(0)
            -------------------------------------------------------------------
            Total size :	6.64MB(0.00B)
            Dynamic size :	6.64MB(0.00B)
            Frozen size :	0.00B(0.00B)
            ===================================================================

        """

        self.in_features = in_features
        self.out_features = out_features
        self.blocks = blocks
        self.init_filters = init_filters

        self.d0_1 = DoubleConvBlock(in_features, init_filters)
        self.d0_2 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        for i in range(1, blocks):
            # conv->relu x2
            layer = DoubleConvBlock(init_filters * (2 ** (i - 1)), init_filters * (2**i))  # fmt: skip
            setattr(self, f"d{i}_1", layer)
            setattr(self, f"d{i}_2", sk.nn.MaxPool2D(kernel_size=2, strides=2))

        self.b0_1 = DoubleConvBlock(init_filters * (2 ** (blocks - 1)), init_filters * (2 ** (blocks)))  # fmt: skip

        for i in range(blocks, 0, -1):
            # upscale and conv to reduce channels size and double row,col size
            layer = UpscaleBlock(init_filters * (2 ** (i)), init_filters * (2 ** (i - 1)))  # fmt: skip
            setattr(self, f"u{i-1}_1", layer)
            layer = DoubleConvBlock(init_filters * (2 ** (i)), init_filters * (2 ** (i - 1)))  # fmt: skip
            setattr(self, f"u{i-1}_3", layer)

        self.f0_1 = sk.nn.Conv2D(init_filters, out_features, kernel_size=1)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
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
        result[f"u{blocks-1}_2"] = _resize_and_cat(result[lhs_key], result[rhs_key])
        result[f"u{blocks-1}_3"] = getattr(self, f"u{blocks-1}_3")(result[f"u{blocks-1}_2"])  # fmt: skip

        for i in range(blocks - 1, 0, -1):
            result[f"u{i-1}_1"] = getattr(self, f"u{i-1}_1")(result[f"u{i}_3"])
            result[f"u{i-1}_2"] = _resize_and_cat(
                result[f"u{i-1}_1"], result[f"d{i-1}_1"]
            )
            result[f"u{i-1}_3"] = getattr(self, f"u{i-1}_3")(result[f"u{i-1}_2"])

        return self.f0_1(result["u0_3"])
