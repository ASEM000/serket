# Copyright 2023 Serket authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from pytreeclass import (
    AtIndexer,
    BaseKey,
    Partial,
    TreeClass,
    bcmap,
    field,
    fields,
    freeze,
    is_frozen,
    is_nondiff,
    is_tree_equal,
    tree_diagram,
    tree_flatten_with_trace,
    tree_graph,
    tree_leaves_with_trace,
    tree_map_with_trace,
    tree_mask,
    tree_mermaid,
    tree_repr,
    tree_repr_with_trace,
    tree_str,
    tree_summary,
    tree_unmask,
    unfreeze,
)

from . import nn
from .nn.evaluation import tree_evaluation
from .nn.state import tree_state

__all__ = (
    # general utils
    "TreeClass",
    "is_tree_equal",
    "field",
    "fields",
    # pprint utils
    "tree_diagram",
    "tree_graph",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_summary",
    # masking utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    "tree_unmask",
    "tree_mask",
    # indexing utils
    "AtIndexer",
    "BaseKey",
    # tree utils
    "bcmap",
    "tree_map_with_trace",
    "tree_leaves_with_trace",
    "tree_flatten_with_trace",
    "tree_repr_with_trace",
    "Partial",
    # serket
    "nn",
    "tree_evaluation",
    "tree_state",
)


__version__ = "0.4.0b1"
