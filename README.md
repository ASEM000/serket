
<div align="center">
<img width="350px" src="assets/serketLogo.svg"></div>

<h2 align="center">The ✨Magical✨ JAX NN Library.</h2>
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

```python
pip install serket
```
**Install development version**
```python
pip install git+https://github.com/ASEM000/serket
```


## 📖 Description<a id="Description"></a>
- `serket` aims to be the most intuitive and easy-to-use Neural network library in JAX.
- `serket` is built on top of [`pytreeclass`](https://github.com/ASEM000/pytreeclass)

- `serket` currently implements 

| Group | Layers |
| ------------- | ------------- |
| Linear  | `Linear`, `Bilinear`,`Identity`   |
|Densely connected|`FNN` (Fully connected network), `PFNN` (Parallel fully connected network)|
| Convolution | `Conv1D`, `Conv2D`, `Conv3D`, `Conv1DTranspose` , `Conv2DTranspose`, `Conv3DTranspose`, `DepthwiseConv1D`, `DepthwiseConv2D`, `DepthwiseConv3D`, `SeparableConv1D`, `SeparableConv2D`, `SeparableConv3D`, `Conv1DLocal`, `Conv2DLocal`, `Conv3DLocal`  |
|Convolution scan |`ConvScan1D`, `ConvScan2D`, `ConvScan3D` (`kernex` backend)(physics-related)|
| Containers| `Sequential`, `Lambda` |
|Pooling|`MaxPool1D`, `MaxPool2D`, `MaxPool3D`, `AvgPool1D`, `AvgPool2D`, `AvgPool3D` `GlobalMaxPool1D`, `GlobalMaxPool2D`, `GlobalMaxPool3D`, `GlobalAvgPool1D`, `GlobalAvgPool2D`, `GlobalAvgPool3D` (`kernex` backend)|
|Reshaping|`Flatten`, `Unflatten`, `FlipLeftRight2D`, `FlipUpDown2D`, `Repeat1D`, `Repeat2D`, `Repeat3D`, `Resize1D`, `Resize2D`, `Resize3D`, `Upsampling1D`, `Upsampling2D`, `Upsampling3D`, `Padding1D`, `Padding2D`, `Padding3D` |
|Crop|`Crop1D`, `Crop2D`, |
|Normalization|`LayerNorm`, `InstanceNorm`, `GroupNorm`|
|Blurring| `AvgBlur2D`, `GaussianBlur2D`|
|Dropout|`Dropout`, `Dropout1D`, `Dropout2D`, `Dropout3D`, |
|Random transforms|`RandomCrop1D`, `RandomCrop2D`, `RandomApply`, `RandomCutout1D`, `RandomCutout2D`, `RandomZoom2D`, `RandomContrast2D` |
|Preprocessing|`HistogramEqualization2D`, `AdjustContrast2D`|
|Activations|`AdaptiveLeakyReLU`,`AdaptiveReLU`,`AdaptiveSigmoid`,`AdaptiveTanh`,<br>`CeLU`,`ELU`,`GELU`,`GLU`<br>,`HardSILU`,`HardShrink`,`HardSigmoid`,`HardSwish`,`HardTanh`,<br>`LeakyReLU`,`LogSigmoid`,`LogSoftmax`,`Mish`,`PReLU`,<br> `ReLU`,`ReLU6`,`SILU`,`SeLU`,`Sigmoid`,`SoftPlus`,`SoftShrink`,<br>`SoftSign`,`Swish`,`Tanh`,`TanhShrink`, `ThresholdedReLU`|
|Finite difference|`Gradient`, `Laplace2D`, `FiniteDiff`, `ParameterizedFiniteDiff` |
|Blocks|`VGG16Block`, `VGG19Block`, `UNetBlock`|

- Other functions:
    1) Finite difference package: `serket.fd`: `serket.fd.fdiff`, `serket.fd.generate_finitediff_coeffs`
    2) Automatic differentiation: `serket.diff`, `serket.diff_and_grad` 


## ⏩ Quick Example: <a id="QuickExample">

<details><summary>Train MNIST</summary>

We will use `tensorflow` datasets for dataloading. for more on interface of jax/tensorflow dataset see [here](https://jax.readthedocs.io/en/latest/notebooks/neural_network_with_tfds_data.html)

```python
# imports
import tensorflow as tf
# Ensure TF does not see GPU and grab all GPU memory.
tf.config.set_visible_devices([], device_type="GPU")
import tensorflow_datasets as tfds
import tensorflow.experimental.numpy as tnp
import jax
import jax.numpy as jnp
import jax.random as jr 
import optax  # for gradient optimization
import serket as sk
import matplotlib.pyplot as plt
import functools as ft
```

```python
# Construct a tf.data.Dataset
batch_size = 128

# convert the samples from integers to floating-point numbers
# and channel first format
def preprocess_data(x):
    # convert to channel first format
    image = tnp.moveaxis(x["image"], -1, 0)
    # normalize to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0

    # one-hot encode the labels
    label = tf.one_hot(x["label"], 10) / 1.0
    return {"image": image, "label": label}


ds_train, ds_test = tfds.load("mnist", split=["train", "test"], shuffle_files=True)
# (batches, batch_size, 1, 28, 28)
ds_train = ds_train.shuffle(1024).map(preprocess_data).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# (batches, 1, 28, 28)
ds_test = ds_test.map(preprocess_data).prefetch(tf.data.AUTOTUNE)
```

### 🏗️ Model definition

We will use `jax.vmap(model)` to apply `model` on batches.
    
```python
@sk.treeclass
class CNN:
    def __init__(self):
        self.conv1 = sk.nn.Conv2D(1, 32, (3, 3), padding="valid")
        self.relu1 = sk.nn.ReLU()
        self.pool1 = sk.nn.MaxPool2D((2, 2), strides=(2, 2))
        self.conv2 = sk.nn.Conv2D(32, 64, (3, 3), padding="valid")
        self.relu2 = sk.nn.ReLU()
        self.pool2 = sk.nn.MaxPool2D((2, 2), strides=(2, 2))
        self.flatten = sk.nn.Flatten(start_dim=0)
        self.dropout = sk.nn.Dropout(0.5)
        self.linear = sk.nn.Linear(5*5*64, 10)

    def __call__(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.dropout(x)
        x = self.linear(x)
        return x

model = CNN()
```

### 🎨 Visualize model
    
<details><summary>Model summary</summary>
    
```python
print(model.summary(show_config=False, array=jnp.empty((1, 28, 28))))  
┌───────┬─────────┬─────────┬──────────────┬─────────────┬─────────────┐
│Name   │Type     │Param #  │Size          │Input        │Output       │
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│conv1  │Conv2D   │320(0)   │1.25KB(0.00B) │f32[1,28,28] │f32[32,26,26]│
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│relu1  │ReLU     │0(0)     │0.00B(0.00B)  │f32[32,26,26]│f32[32,26,26]│
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│pool1  │MaxPool2D│0(0)     │0.00B(0.00B)  │f32[32,26,26]│f32[32,13,13]│
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│conv2  │Conv2D   │18,496(0)│72.25KB(0.00B)│f32[32,13,13]│f32[64,11,11]│
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│relu2  │ReLU     │0(0)     │0.00B(0.00B)  │f32[64,11,11]│f32[64,11,11]│
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│pool2  │MaxPool2D│0(0)     │0.00B(0.00B)  │f32[64,11,11]│f32[64,5,5]  │
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│flatten│Flatten  │0(0)     │0.00B(0.00B)  │f32[64,5,5]  │f32[1600]    │
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│dropout│Dropout  │0(0)     │0.00B(0.00B)  │f32[1600]    │f32[1600]    │
├───────┼─────────┼─────────┼──────────────┼─────────────┼─────────────┤
│linear │Linear   │16,010(0)│62.54KB(0.00B)│f32[1600]    │f32[10]      │
└───────┴─────────┴─────────┴──────────────┴─────────────┴─────────────┘
Total count :	34,826(0)
Dynamic count :	34,826(0)
Frozen count :	0(0)
------------------------------------------------------------------------
Total size :	136.04KB(0.00B)
Dynamic size :	136.04KB(0.00B)
Frozen size :	0.00B(0.00B)
========================================================================
```
        
</details>

<details><summary>tree diagram </summary>
    
```python
print(model.tree_diagram())
CNN
    ├── conv1=Conv2D
    │   ├── weight=f32[32,1,3,3]
    │   ├── bias=f32[32,1,1]
    │   ├*─ in_features=1
    │   ├*─ out_features=32
    │   ├*─ kernel_size=(3, 3)
    │   ├*─ strides=(1, 1)
    │   ├*─ padding=((0, 0), (0, 0))
    │   ├*─ input_dilation=(1, 1)
    │   ├*─ kernel_dilation=(1, 1)
    │   ├── weight_init_func=Partial(init(key,shape,dtype))
    │   ├── bias_init_func=Partial(zeros(key,shape,dtype))
    │   └*─ groups=1    
    ├── relu1=ReLU  
    ├*─ pool1=MaxPool2D
    │   ├*─ kernel_size=(2, 2)
    │   ├*─ strides=(2, 2)
    │   └*─ padding='valid' 
    ├── conv2=Conv2D
    │   ├── weight=f32[64,32,3,3]
    │   ├── bias=f32[64,1,1]
    │   ├*─ in_features=32
    │   ├*─ out_features=64
    │   ├*─ kernel_size=(3, 3)
    │   ├*─ strides=(1, 1)
    │   ├*─ padding=((0, 0), (0, 0))
    │   ├*─ input_dilation=(1, 1)
    │   ├*─ kernel_dilation=(1, 1)
    │   ├── weight_init_func=Partial(init(key,shape,dtype))
    │   ├── bias_init_func=Partial(zeros(key,shape,dtype))
    │   └*─ groups=1    
    ├── relu2=ReLU  
    ├*─ pool2=MaxPool2D
    │   ├*─ kernel_size=(2, 2)
    │   ├*─ strides=(2, 2)
    │   └*─ padding='valid' 
    ├*─ flatten=Flatten
    │   ├*─ start_dim=0
    │   └*─ end_dim=-1  
    ├── dropout=Dropout
    │   ├*─ p=0.5
    │   └── eval=None   
    └── linear=Linear
        ├── weight=f32[1600,10]
        ├── bias=f32[10]
        ├*─ in_features=1600
        └*─ out_features=10  
    
 ```
    
</details>
    
<details><summary>Plot sample predictions before training</summary>
    
```python
 
# set all dropout off
test_model = model.at[model == "eval"].set(True, is_leaf=lambda x: x is None)

def show_images_with_predictions(model, images, one_hot_labels):
    logits = jax.vmap(model)(images)
    predictions = jnp.argmax(logits, axis=-1)
    fig, axes = plt.subplots(5, 5, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].reshape(28, 28), cmap="binary")
        ax.set(title=f"Prediction: {predictions[i]}\nLabel: {jnp.argmax(one_hot_labels[i], axis=-1)}")
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

example = ds_test.take(25).as_numpy_iterator()
example = list(example)
sample_test_images = jnp.stack([x["image"] for x in example])
sample_test_labels = jnp.stack([x["label"] for x in example])

show_images_with_predictions(test_model, sample_test_images, sample_test_labels)
```
![image](assets/before_training.svg)
 
</details>
    
### 🏃 Train the model

```python
@ft.partial(jax.value_and_grad, has_aux=True)
def loss_func(model, batched_images, batched_one_hot_labels):
    logits = jax.vmap(model)(batched_images)
    loss = jnp.mean(optax.softmax_cross_entropy(logits=logits, labels=batched_one_hot_labels))
    return loss, logits


# using optax for gradient updates
optim = optax.adam(1e-3)
optim_state = optim.init(model)


@jax.jit
def batch_step(model, batched_images, batched_one_hot_labels, optim_state):
    (loss, logits), grads = loss_func(model, batched_images, batched_one_hot_labels)
    updates, optim_state = optim.update(grads, optim_state)
    model = optax.apply_updates(model, updates)
    accuracy = jnp.mean(jnp.argmax(logits, axis=-1) == jnp.argmax(batched_one_hot_labels, axis=-1))
    return model, optim_state, loss, accuracy


epochs = 5

for i in range(epochs):
    epoch_accuracy = []
    epoch_loss = []

    for example in ds_train.as_numpy_iterator():
        image, label = example["image"], example["label"]
        model, optim_state, loss, accuracy = batch_step(model, image, label, optim_state)
        epoch_accuracy.append(accuracy)
        epoch_loss.append(loss)

    epoch_loss = jnp.mean(jnp.array(epoch_loss))
    epoch_accuracy = jnp.mean(jnp.array(epoch_accuracy))

    print(f"epoch:{i+1:00d}\tloss:{epoch_loss:.4f}\taccuracy:{epoch_accuracy:.4f}")
    
# epoch:1	loss:0.2706	accuracy:0.9268
# epoch:2	loss:0.0725	accuracy:0.9784
# epoch:3	loss:0.0533	accuracy:0.9836
# epoch:4	loss:0.0442	accuracy:0.9868
# epoch:5	loss:0.0368	accuracy:0.9889
```
    
    
### 🎨 Visualize After training

```python
test_model = model.at[model == "eval"].set(True, is_leaf=lambda x: x is None)
show_images_with_predictions(test_model, sample_test_images, sample_test_labels)
```

<details> 
    
![image](assets/after_training.svg)

</details>

</details>

## 🥶 Freezing parameters /Fine tuning<a id="Freezing" >

✨[See here for more about freezing](https://github.com/ASEM000/PyTreeClass#%EF%B8%8F-model-surgery)✨

## 🔘 Filtering by masking<a id="Filtering" >
✨[See here for more about filterning ](https://github.com/ASEM000/PyTreeClass#%EF%B8%8F-filtering-with-at-)✨
