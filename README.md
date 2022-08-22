
<div align="center">
<img width="350px" src="assets/serketLogo.svg"></div>
<h2 align="center">The JAX NN Library.</h2>


![Tests](https://github.com/ASEM000/serket/actions/workflows/tests.yml/badge.svg)
![pyver](https://img.shields.io/badge/python-3.7%203.8%203.9%203.10-red)
![codestyle](https://img.shields.io/badge/codestyle-black-lightgrey)
[![codecov](https://codecov.io/gh/ASEM000/serket/branch/main/graph/badge.svg?token=C6NXOK9EVS)](https://codecov.io/gh/ASEM000/serket)

## 🛠️ Installation<a id="Installation"></a>

```python
pip install serket
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

### Model definition
```python

>>> model = Sequential([
    Linear(1,128),
    Lambda(jax.nn.relu),
    Linear(128,128),
    Lambda(jax.nn.relu),
    Linear(128,1),
])

>>> print(model.tree_diagram())
Sequential
    └── layers=<class 'list'>
        ├── layers_0=Linear
        │   ├── weight=f32[1,128]
        │   ├── bias=f32[128]
        │   ├*─ in_features=1
        │   ├*─ out_features=128
        │   ├*─ weight_init_func=init(key,shape,dtype)
        │   └*─ bias_init_func=Lambda(key,shape)    
        ├── layers_1=Lambda
        │   └*─ func=relu(*args,**kwargs)   
        ├── layers_2=Linear
        │   ├── weight=f32[128,128]
        │   ├── bias=f32[128]
        │   ├*─ in_features=128
        │   ├*─ out_features=128
        │   ├*─ weight_init_func=init(key,shape,dtype)
        │   └*─ bias_init_func=Lambda(key,shape)    
        ├── layers_3=Lambda
        │   └*─ func=relu(*args,**kwargs)   
        └── layers_4=Linear
            ├── weight=f32[128,1]
            ├── bias=f32[1]
            ├*─ in_features=128
            ├*─ out_features=1
            ├*─ weight_init_func=init(key,shape,dtype)
            └*─ bias_init_func=Lambda(key,shape)
```

```python
>>> print(model.summary())
┌───────────────┬───────────┬─────────┬───────┬───────────────────┐
│Name           │Type       │Param #  │Size   │Config             │
├───────────────┼───────────┼─────────┼───────┼───────────────────┤
│layers/layers_0│list/Linear│256(0)   │1.00KB │weight=f32[1,128]  │
│               │           │         │(0.00B)│bias=f32[128]      │
├───────────────┼───────────┼─────────┼───────┼───────────────────┤
│layers/layers_1│list/Lambda│0(0)     │0.00B  │                   │
│               │           │         │(0.00B)│                   │
├───────────────┼───────────┼─────────┼───────┼───────────────────┤
│layers/layers_2│list/Linear│16,512(0)│64.50KB│weight=f32[128,128]│
│               │           │         │(0.00B)│bias=f32[128]      │
├───────────────┼───────────┼─────────┼───────┼───────────────────┤
│layers/layers_3│list/Lambda│0(0)     │0.00B  │                   │
│               │           │         │(0.00B)│                   │
├───────────────┼───────────┼─────────┼───────┼───────────────────┤
│layers/layers_4│list/Linear│129(0)   │516.00B│weight=f32[128,1]  │
│               │           │         │(0.00B)│bias=f32[1]        │
└───────────────┴───────────┴─────────┴───────┴───────────────────┘
Total count :	16,897(0)
Dynamic count :	16,897(0)
Frozen count :	0(0)
-------------------------------------------------------------------
Total size :	66.00KB(0.00B)
Dynamic size :	66.00KB(0.00B)
Frozen size :	0.00B(0.00B)
===================================================================

```

### Train
```python
x = jnp.linspace(0,1,100)[:,None]
y = x**3 + jax.random.uniform(jax.random.PRNGKey(0),(100,1))*0.01

@jax.value_and_grad
def loss_func(model,x,y):
    return jnp.mean((model(x)-y)**2)

@jax.jit
def update(model,x,y):
    value,grad = loss_func(model,x,y)
    return value , model - 1e-3*grad

for _ in range(20_000):
    value,model = update(model,x,y)
```

### Filter
- Filter by (1)value, (2)`field` name, (3)`field` type, (4)`field` metadata
- See [here](https://github.com/ASEM000/PyTreeClass#%EF%B8%8F-filtering-with-at-) for more
