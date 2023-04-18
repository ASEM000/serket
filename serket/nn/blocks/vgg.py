from __future__ import annotations

import jax
import jax.random as jr
import pytreeclass as pytc

import serket as sk


class VGG16Block(pytc.TreeClass):
    def __init__(
        self,
        in_features: int,
        *,
        pooling: str = "max",
        key: jr.KeyArray = jr.PRNGKey(0),
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

        Example:
            >>> print(model.summary(compact=True, array=jnp.ones([3,320,320])))
            ┌─────────┬───────────────┬────────────┬───────────────┬────────────────┬────────────────┐
            │Name     │Type           │Param #     │Size           │Input           │Output          │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_1_1 │Conv2D         │1,792(0)    │7.00KB(0.00B)  │f32[3,320,320]  │f32[64,320,320] │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_1_2 │Conv2D         │36,928(0)   │144.25KB(0.00B)│f32[64,320,320] │f32[64,320,320] │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │maxpool_1│MaxPool2D      │0(0)        │0.00B(0.00B)   │f32[64,320,320] │f32[64,160,160] │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_2_1 │Conv2D         │73,856(0)   │288.50KB(0.00B)│f32[64,160,160] │f32[128,160,160]│
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_2_2 │Conv2D         │147,584(0)  │576.50KB(0.00B)│f32[128,160,160]│f32[128,160,160]│
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │maxpool_2│MaxPool2D      │0(0)        │0.00B(0.00B)   │f32[128,160,160]│f32[128,80,80]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_3_1 │Conv2D         │295,168(0)  │1.13MB(0.00B)  │f32[128,80,80]  │f32[256,80,80]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_3_2 │Conv2D         │590,080(0)  │2.25MB(0.00B)  │f32[256,80,80]  │f32[256,80,80]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_3_3 │Conv2D         │590,080(0)  │2.25MB(0.00B)  │f32[256,80,80]  │f32[256,80,80]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │maxpool_3│MaxPool2D      │0(0)        │0.00B(0.00B)   │f32[256,80,80]  │f32[256,40,40]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_4_1 │Conv2D         │1,180,160(0)│4.50MB(0.00B)  │f32[256,40,40]  │f32[512,40,40]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_4_2 │Conv2D         │2,359,808(0)│9.00MB(0.00B)  │f32[512,40,40]  │f32[512,40,40]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_4_3 │Conv2D         │2,359,808(0)│9.00MB(0.00B)  │f32[512,40,40]  │f32[512,40,40]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │maxpool_4│MaxPool2D      │0(0)        │0.00B(0.00B)   │f32[512,40,40]  │f32[512,20,20]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_5_1 │Conv2D         │2,359,808(0)│9.00MB(0.00B)  │f32[512,20,20]  │f32[512,20,20]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_5_2 │Conv2D         │2,359,808(0)│9.00MB(0.00B)  │f32[512,20,20]  │f32[512,20,20]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │conv_5_3 │Conv2D         │2,359,808(0)│9.00MB(0.00B)  │f32[512,20,20]  │f32[512,20,20]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │maxpool_5│MaxPool2D      │0(0)        │0.00B(0.00B)   │f32[512,20,20]  │f32[512,10,10]  │
            ├─────────┼───────────────┼────────────┼───────────────┼────────────────┼────────────────┤
            │pooling  │GlobalMaxPool2D│0(0)        │0.00B(0.00B)   │f32[512,10,10]  │f32[512,1,1]    │
            └─────────┴───────────────┴────────────┴───────────────┴────────────────┴────────────────┘
            Total count :	14,714,688(0)
            Dynamic count :	14,714,688(0)
            Frozen count :	0(0)
            ------------------------------------------------------------------------------------------
            Total size :	56.13MB(0.00B)
            Dynamic size :	56.13MB(0.00B)
            Frozen size :	0.00B(0.00B)
            ==========================================================================================
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

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
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


class VGG19Block(pytc.TreeClass):
    def __init__(
        self,
        in_feautres: int,
        *,
        pooling: str = "max",
        key: jr.KeyArray = jr.PRNGKey(0),
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

    def __call__(self, x: jax.Array, **kwargs) -> jax.Array:
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
