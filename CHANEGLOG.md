# Changelog

## V0.x

### Changes

- `ScanRNN` changes:

  - `cell.init_state` is deprecated use `sk.tree_state(cell, ...)` instead.

- Naming changes:
  - `***_init_func` -> `***_init` shorter and more concise
  - `gamma_init_func` -> `weight_init`
  - `beta_init_func` -> `bias_init`
  - `act_func` -> `act`
- `MLP` produces smaller `jaxprs` and are faster to compile. for my use case -higher order differentiation through `PINN`- the new `MLP` is faster to compile.
- `kernel_dilation` -> `dilation`
- `input_dilation` -> Removed.
- `p` -> `drop_rate` in all dropout layers
- `FlipLeftRight2D` -> `HorizontalFlip2D`
- `FlipUpDown2D` -> `VerticalFlip2D`

### Additions

- `tree_eval`: a dispatcher to define layers evaluation rule. for example `Dropout` is changed to `Identity` when `tree_eval` is applied.

  ```python
  @sk.tree_eval.def_eval(sk.nn.Dropout)
  def dropout_evaluation(_) -> sk.nn.Identity:
      return sk.nn.Identity()
  ```

- `tree_state`: a dispatcher to define state intialization for `BatchNorm`, `RNN` cells.

  ```python
  @sk.tree_state.def_state(sk.nn.SimpleRNNCell)
  def simple_rnn_init_state(cell: SimpleRNNCell, array: jax.Array  | None) -> SimpleRNNState:
      del array  # unused
      return SimpleRNNState(jnp.zeros([cell.hidden_features]))
  ```

- `MultiHeadAttention`
- `BatchNorm`
- `RandomHorizontalShear2D`
- `RandomPerspective2D`
- `RandomRotate2D`
- `RandomVerticalShear2D`
- `Rotate2D`
- `VerticalShear2D`
- `Pixelate2D`
- `Solarize2D`
- `Posterize2D`
- `JigSaw2D`

### Deprecations

- `Bilinear` is deprecated, use `Multilinear((in1_features, in2_features), out_features)`
- `HistogramEqualization2D`
- Remove `.blocks`, and will move it to examples
