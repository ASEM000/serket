# Changelog

`ScanRNN` changes:

- `backward_cell=...` is deprecated , instead use `ScanRNN(forward_cell, backward_cell, reverse=(False,True))`
- `ScanRNN` now accepts arbitrary number of cells as input and an argument `reverse` to decide whether to reverse the corresponding cell or not.
- `return_state` is added to control whether the final carry is returned or not.
- `cell.init_state` is deprecated use `sk.tree_state(cell, ...)` instead.
