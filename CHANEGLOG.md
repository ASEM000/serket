# Changelog

## V0.x

### Changes

- `ScanRNN` changes:

  - `backward_cell=...` is deprecated , instead use `ScanRNN(forward_cell, backward_cell, reverse=(False,True))`
  - `ScanRNN` now accepts arbitrary number of cells as input and an argument `reverse` to decide whether to reverse the corresponding cell or not.
  - `return_state` is added to control whether the final carry is returned or not.
  - `cell.init_state` is deprecated use `sk.tree_state(cell, ...)` instead.

- Naming changes:
  - `***_init_func` -> `***_init` shorter and more concise
  - `gamma_init_func` -> `weight_init`
  - `beta_init_func` -> `bias_init`
- `MLP` produces smaller `jaxprs` and are faster to compile. for my use case -higher order differentiation through `PINN`- the new `MLP` is faster to compile.
- `kernel_dilation` -> `dilation`
- `input_dilation` -> Removed.

### Additions

- `tree_eval`: a dispatcher to define layers evaluation rule. for example `Dropout` is changed to `Identity` when `tree_state` is applied.

  ```python
  @sk.tree_eval.def_eval(sk.nn.Dropout)
  def dropout_evaluation(_) -> sk.nn.Identity:
      return sk.nn.Identity()
  ```

- `tree_state`: a dispatcher to define state intialization for `BatchNorm`, `RNN` cells.

  ```python
  @sk.tree_state.def_state(sk.nn.SimpleRNNCell)
  def simple_rnn_init_state(cell: SimpleRNNCell) -> SimpleRNNState:
      return SimpleRNNState(jnp.zeros([cell.hidden_features]))
  ```

### Deprecations

- `Bilinear` is deprecated, use `Multilinear((in1_features, in2_features), out_features)`
