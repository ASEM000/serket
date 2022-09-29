from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
import pytreeclass as pytc

import serket as sk


@pytc.treeclass
class VGG16Block:
    def __init__(
        self, in_features: int, *, pooling: str = "max", key: jr.PRNGKey = jr.PRNGKey(0)
    ):
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
        keys = jr.split(key, 13)

        # block 1
        self.conv_1_1 = sk.nn.Conv2D(in_features, 64, 3, padding="same", key=keys[0])
        self.conv_1_2 = sk.nn.Conv2D(64, 64, 3, padding="same", key=keys[1])
        self.maxpool_1 = sk.nn.MaxPool2D(2, strides=2)

        # block 2
        self.conv_2_1 = sk.nn.Conv2D(64, 128, 3, padding="same", key=keys[2])
        self.conv_2_2 = sk.nn.Conv2D(128, 128, 3, padding="same", key=keys[3])
        self.maxpool_2 = sk.nn.MaxPool2D(2, strides=2)

        # block 3
        self.conv_3_1 = sk.nn.Conv2D(128, 256, 3, padding="same", key=keys[4])
        self.conv_3_2 = sk.nn.Conv2D(256, 256, 3, padding="same", key=keys[5])
        self.conv_3_3 = sk.nn.Conv2D(256, 256, 3, padding="same", key=keys[6])
        self.maxpool_3 = sk.nn.MaxPool2D(2, strides=2)

        # block 4
        self.conv_4_1 = sk.nn.Conv2D(256, 512, 3, padding="same", key=keys[7])
        self.conv_4_2 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[8])
        self.conv_4_3 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[9])
        self.maxpool_4 = sk.nn.MaxPool2D(2, strides=2)

        # block 5
        self.conv_5_1 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[10])
        self.conv_5_2 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[11])
        self.conv_5_3 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[12])
        self.maxpool_5 = sk.nn.MaxPool2D(2, strides=2)

        self.pooling = sk.nn.GlobalMaxPool2D() if pooling == "max" else sk.nn.GlobalAvgPool2D()  # fmt: skip

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
    def __init__(
        self, in_feautres: int, *, pooling: str = "max", key: jr.PRNGKey = jr.PRNGKey(0)
    ):
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
        keys = jr.split(jr.PRNGKey(0), 16)

        # block 1
        self.conv_1_1 = sk.nn.Conv2D(in_feautres, 64, 3, padding="same", key=keys[0])
        self.conv_1_2 = sk.nn.Conv2D(64, 64, 3, padding="same", key=keys[1])
        self.maxpool_1 = sk.nn.MaxPool2D(2, strides=2)

        # block 2
        self.conv_2_1 = sk.nn.Conv2D(64, 128, 3, padding="same", key=keys[2])
        self.conv_2_2 = sk.nn.Conv2D(128, 128, 3, padding="same", key=keys[3])
        self.maxpool_2 = sk.nn.MaxPool2D(2, strides=2)

        # block 3
        self.conv_3_1 = sk.nn.Conv2D(128, 256, 3, padding="same", key=keys[4])
        self.conv_3_2 = sk.nn.Conv2D(256, 256, 3, padding="same", key=keys[5])
        self.conv_3_3 = sk.nn.Conv2D(256, 256, 3, padding="same", key=keys[6])
        self.conv_3_4 = sk.nn.Conv2D(256, 256, 3, padding="same", key=keys[7])
        self.maxpool_3 = sk.nn.MaxPool2D(2, strides=2)

        # block 4
        self.conv_4_1 = sk.nn.Conv2D(256, 512, 3, padding="same", key=keys[8])
        self.conv_4_2 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[9])
        self.conv_4_3 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[10])
        self.conv_4_4 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[11])
        self.maxpool_4 = sk.nn.MaxPool2D(2, strides=2)

        # block 5
        self.conv_5_1 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[12])
        self.conv_5_2 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[13])
        self.conv_5_3 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[14])
        self.conv_5_4 = sk.nn.Conv2D(512, 512, 3, padding="same", key=keys[15])
        self.maxpool_5 = sk.nn.MaxPool2D(2, strides=2)

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
