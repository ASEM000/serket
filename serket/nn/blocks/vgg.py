from __future__ import annotations

import jax
import jax.numpy as jnp
import pytreeclass as pytc

import serket as sk


@pytc.treeclass
class VGG16Block:
    def __init__(self, in_features: int, pooling: str = "max"):
        """
        Args:
            in_features: number of input features
            pooling: pooling method to use. GlobalMaxPool2D(`max`) or GlobalAvgPool2D(`avg`).

        Note:
            if num_classes is None, then the classifier is not added.
            see:
                https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/vgg16.py
                https://arxiv.org/abs/1409.1556
        """

        # block 1
        self.conv_1_1 = sk.nn.Conv2D(
            in_features,
            64,
            kernel_size=3,
            padding="same",
        )
        self.conv_1_2 = sk.nn.Conv2D(64, 64, kernel_size=3, padding="same")
        self.maxpool_1 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 2
        self.conv_2_1 = sk.nn.Conv2D(64, 128, kernel_size=3, padding="same")
        self.conv_2_2 = sk.nn.Conv2D(128, 128, kernel_size=3, padding="same")
        self.maxpool_2 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 3
        self.conv_3_1 = sk.nn.Conv2D(128, 256, kernel_size=3, padding="same")
        self.conv_3_2 = sk.nn.Conv2D(256, 256, kernel_size=3, padding="same")
        self.conv_3_3 = sk.nn.Conv2D(256, 256, kernel_size=3, padding="same")
        self.maxpool_3 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 4
        self.conv_4_1 = sk.nn.Conv2D(256, 512, kernel_size=3, padding="same")
        self.conv_4_2 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_4_3 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.maxpool_4 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 5
        self.conv_5_1 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_5_2 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_5_3 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.maxpool_5 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        self.pooling = (
            sk.nn.GlobalMaxPool2D() if pooling == "max" else sk.nn.GlobalAvgPool2D()
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.conv_1_1(x)
        x = jax.nn.relu(x)
        x = self.conv_1_2(x)
        x = jax.nn.relu(x)
        x = self.maxpool_1(x)

        x = self.conv_2_1(x)
        x = jax.nn.relu(x)
        x = self.conv_2_2(x)
        x = jax.nn.relu(x)
        x = self.maxpool_2(x)

        x = self.conv_3_1(x)
        x = jax.nn.relu(x)
        x = self.conv_3_2(x)
        x = jax.nn.relu(x)
        x = self.conv_3_3(x)
        x = jax.nn.relu(x)
        x = self.maxpool_3(x)

        x = self.conv_4_1(x)
        x = jax.nn.relu(x)
        x = self.conv_4_2(x)
        x = jax.nn.relu(x)
        x = self.conv_4_3(x)
        x = jax.nn.relu(x)
        x = self.maxpool_4(x)

        x = self.conv_5_1(x)
        x = jax.nn.relu(x)
        x = self.conv_5_2(x)
        x = jax.nn.relu(x)
        x = self.conv_5_3(x)
        x = jax.nn.relu(x)
        x = self.maxpool_5(x)
        x = self.pooling(x)
        return x


@pytc.treeclass
class VGG19Block:
    def __init__(self, in_feautres: int, pooling: str = "max"):
        """
        Args:
            in_features: number of input features
            pooling: pooling method to use. GlobalMaxPool2D(`max`) or GlobalAvgPool2D(`avg`).

        Note:
            if num_classes is None, then the classifier is not added.
            see:
                https://github.com/keras-team/keras/blob/v2.10.0/keras/applications/vgg19.py
                https://arxiv.org/abs/1409.1556
        """
        # block 1
        self.conv_1_1 = sk.nn.Conv2D(in_feautres, 64, kernel_size=3, padding="same")
        self.conv_1_2 = sk.nn.Conv2D(64, 64, kernel_size=3, padding="same")
        self.maxpool_1 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 2
        self.conv_2_1 = sk.nn.Conv2D(64, 128, kernel_size=3, padding="same")
        self.conv_2_2 = sk.nn.Conv2D(128, 128, kernel_size=3, padding="same")
        self.maxpool_2 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 3
        self.conv_3_1 = sk.nn.Conv2D(128, 256, kernel_size=3, padding="same")
        self.conv_3_2 = sk.nn.Conv2D(256, 256, kernel_size=3, padding="same")
        self.conv_3_3 = sk.nn.Conv2D(256, 256, kernel_size=3, padding="same")
        self.conv_3_4 = sk.nn.Conv2D(256, 256, kernel_size=3, padding="same")
        self.maxpool_3 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 4
        self.conv_4_1 = sk.nn.Conv2D(256, 512, kernel_size=3, padding="same")
        self.conv_4_2 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_4_3 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_4_4 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.maxpool_4 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        # block 5
        self.conv_5_1 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_5_2 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_5_3 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.conv_5_4 = sk.nn.Conv2D(512, 512, kernel_size=3, padding="same")
        self.maxpool_5 = sk.nn.MaxPool2D(kernel_size=2, strides=2)

        self.pooling = (
            sk.nn.GlobalMaxPool2D() if pooling == "max" else sk.nn.GlobalAvgPool2D()
        )

    def __call__(self, x: jnp.ndarray, **kwargs) -> jnp.ndarray:
        x = self.conv_1_1(x)
        x = jax.nn.relu(x)
        x = self.conv_1_2(x)
        x = jax.nn.relu(x)
        x = self.maxpool_1(x)

        x = self.conv_2_1(x)
        x = jax.nn.relu(x)
        x = self.conv_2_2(x)
        x = jax.nn.relu(x)
        x = self.maxpool_2(x)

        x = self.conv_3_1(x)
        x = jax.nn.relu(x)
        x = self.conv_3_2(x)
        x = jax.nn.relu(x)
        x = self.conv_3_3(x)
        x = jax.nn.relu(x)
        x = self.conv_3_4(x)
        x = jax.nn.relu(x)
        x = self.maxpool_3(x)

        x = self.conv_4_1(x)
        x = jax.nn.relu(x)
        x = self.conv_4_2(x)
        x = jax.nn.relu(x)
        x = self.conv_4_3(x)
        x = jax.nn.relu(x)
        x = self.conv_4_4(x)
        x = jax.nn.relu(x)

        x = self.conv_5_1(x)
        x = jax.nn.relu(x)
        x = self.conv_5_2(x)
        x = jax.nn.relu(x)
        x = self.conv_5_3(x)
        x = jax.nn.relu(x)
        x = self.conv_5_4(x)
        x = jax.nn.relu(x)
        x = self.maxpool_5(x)

        x = self.pooling(x)
        return x
