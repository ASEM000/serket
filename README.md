
<div align="center">
<img width="350px" src="assets/serketLogo.svg"></div>

<h2 align="center">The ✨Magical✨ JAX NN Library.</h2>
<h5 align = "center"> *Serket is the goddess of magic in Egyptian mythology </h5>

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
  - `Linear`, `FNN`
  - `Dropout`
  - `Sequential`
  - `Lambda`


## ⏩ Quick Example <a id="QuickExample">

Simple Fully connected neural network.

### 🏗️ Model definition
```python
import serket as sk 
import jax
import jax.numpy as jnp
import jax.random as jr

@sk.treeclass
class NN:
    def __init__(
        self, 
        in_features:int, 
        out_features:int, 
        hidden_features: int, key:jr.PRNGKey = jr.PRNGKey(0)):

        k1,k2,k3 = jr.split(key, 3)

        self.l1 = sk.nn.Linear(in_features, hidden_features, key=k1)
        self.l2 = sk.nn.Linear(hidden_features, hidden_features, key=k2)
        self.l3 = sk.nn.Linear(hidden_features, out_features, key=k3)
    
    def __call__(self, x):
        x = self.l1(x)
        x = jax.nn.relu(x)
        x = self.l2(x)
        x = jax.nn.relu(x)
        x = self.l3(x)
        return x

model = NN(
    in_features=1, 
    out_features=1, 
    hidden_features=128, 
    key=jr.PRNGKey(0))
```
### 🎨 Visualize

#### Tree diagram
```python
# `*` represents untrainable(static) nodes.
print(model.tree_diagram())
NN
    ├── l1=Linear
    │   ├── weight=f32[1,128]
    │   ├── bias=f32[128]
    │   ├*─ in_features=1
    │   ├*─ out_features=128
    │   ├*─ weight_init_func=init(key,shape,dtype)
    │   └*─ bias_init_func=Lambda(key,shape)    
    ├── l2=Linear
    │   ├── weight=f32[128,128]
    │   ├── bias=f32[128]
    │   ├*─ in_features=128
    │   ├*─ out_features=128
    │   ├*─ weight_init_func=init(key,shape,dtype)
    │   └*─ bias_init_func=Lambda(key,shape)    
    └── l3=Linear
        ├── weight=f32[128,1]
        ├── bias=f32[1]
        ├*─ in_features=128
        ├*─ out_features=1
        ├*─ weight_init_func=init(key,shape,dtype)
        └*─ bias_init_func=Lambda(key,shape) 
```

#### Tree summary
```python
>>> print(model.summary())
┌────┬──────┬─────────┬───────┬───────────────────┐
│Name│Type  │Param #  │Size   │Config             │
├────┼──────┼─────────┼───────┼───────────────────┤
│l1  │Linear│256(0)   │1.00KB │weight=f32[1,128]  │
│    │      │         │(0.00B)│bias=f32[128]      │
├────┼──────┼─────────┼───────┼───────────────────┤
│l2  │Linear│16,512(0)│64.50KB│weight=f32[128,128]│
│    │      │         │(0.00B)│bias=f32[128]      │
├────┼──────┼─────────┼───────┼───────────────────┤
│l3  │Linear│129(0)   │516.00B│weight=f32[128,1]  │
│    │      │         │(0.00B)│bias=f32[1]        │
└────┴──────┴─────────┴───────┴───────────────────┘
Total count :	16,897(0)
Dynamic count :	16,897(0)
Frozen count :	0(0)
---------------------------------------------------
Total size :	66.00KB(0.00B)
Dynamic size :	66.00KB(0.00B)
Frozen size :	0.00B(0.00B)
===================================================
```

### ‍🔧 Train
```python
import matplotlib.pyplot as plt

x = jnp.linspace(0,1,100)[:,None]
y = x**3 + jax.random.uniform(jax.random.PRNGKey(0),(100,1))*0.01


@jax.value_and_grad
def loss_func(model,x,y):
    return jnp.mean((model(x)-y)**2)

@jax.jit
def update(model,x,y):
    value,grad = loss_func(model,x,y)
    return value , model - 1e-3*grad

plt.plot(x,y,'-k',label='True')
plt.plot(x,model(x),'-r',label='Prediction')
plt.title("Before training")
plt.legend()
plt.show()

for _ in range(20_000):
    value,model = update(model,x,y)

plt.plot(x,y,'-k',label='True')
plt.plot(x,model(x),'-r',label='Prediction')
plt.title("After training")
plt.legend()
plt.show()
```

<div align = "center" >
<table><tr>
<td><div align = "center" > <img width = "350px" src= "assets/before_training.svg" ></div></td>
<td><div align = "center" > <img width = "350px" src= "assets/after_training.svg" ></div></td>
</tr>
</table>
</div>

## 🥶 Freezing parameters /Fine tuning<a id="Freezing" >
In `serket` simply use `.freeze()`/`.unfreeze()` on `treeclass` instance to freeze/unfreeze it is parameters.
```python
# Freeze the entire model
frozen_model = model.freeze()

@jax.value_and_grad
def loss_func(model,x,y):
    return jnp.mean((model(x)-y)**2)

@jax.jit
def update(model,x,y):
    value,grad = loss_func(model,x,y)
    return value , model - 1e-3*grad

plt.plot(x,y,'-k',label='True')
plt.plot(x,frozen_model(x),'-r',label='Prediction')
plt.title("Before training")
plt.legend()

plt.show()
for _ in range(20_000):
    value,frozen_model = update(frozen_model,x,y)

plt.plot(x,y,'-k',label='True')
plt.plot(x,frozen_model(x),'-r',label='Prediction')
plt.title("After training")
plt.legend()
plt.show()
```

<div align="center">
<table><tr>
<td><div align = "center" > <img width = "350px" src= "assets/frozen_before_training.svg" ></div></td>
<td><div align = "center" > <img width = "350px" src= "assets/frozen_after_training.svg" ></div></td>
</tr>
</table>
</div>


## 🔘 Filtering by masking<a id="Filterning" >

### Filter by value
```python
# get model negative values
negative_model = model.at[model<0].get()

# Set negative values to 0
zeroed_model = model.at[model<0].set(0)

# Apply `jnp.cos` to negative values
cosined_model = model.at[model<0].apply(jnp.cos)
```
### Filter by field name 
```python
# get model layer named `l1`
l1_model = model.at[model == "l1" ].get()

# Set `l1` values to 0
zeroed_model = model.at[model == "l1" ].set(0)

# Apply `jnp.cos` to `l1` 
cosined_model = model.at[model == "l1" ].apply(jnp.cos)
```

### Filter by field type
```python
# get all model `Linear` layers
l1_model = model.at[model == sk.nn.Linear ].get()

# Set `Linear` layers to 0
zeroed_model = model.at[model == sk.nn.Linear ].set(0)

# Apply `jnp.cos` to all `Linear` layers 
cosined_model = model.at[model == sk.nn.Linear ].apply(jnp.cos)
```

### Filter by mixed masks
```
# Set all `Linear` bias to 0
mask = (model == sk.nn.Linear) & (model == "bias" )
zero_bias_model = model.at[mask].set(0.)
```

✨[See here for more about filterning ](https://github.com/ASEM000/PyTreeClass#%EF%B8%8F-filtering-with-at-)✨
