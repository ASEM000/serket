# Copyright 2023 serket authors
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
    autoinit,
    bcmap,
    field,
    fields,
    freeze,
    is_frozen,
    is_nondiff,
    is_tree_equal,
    leafwise,
    tree_diagram,
    tree_graph,
    tree_mask,
    tree_mermaid,
    tree_repr,
    tree_str,
    tree_summary,
    tree_unmask,
    unfreeze,
)

from serket._src.custom_transform import tree_eval, tree_state
from serket._src.nn.activation import def_act_entry
from serket._src.nn.initialization import def_init_entry

from . import cluster, image, nn

__all__ = [
    # general utils
    "TreeClass",
    "is_tree_equal",
    "field",
    "fields",
    "autoinit",
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
    "Partial",
    "leafwise",
    # serket
    "cluster",
    "nn",
    "image",
    "tree_eval",
    "tree_state",
    "def_init_entry",
    "def_act_entry",
]


__version__ = "0.2.0rc3"
