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
    TreeClass,
    field,
    freeze,
    is_frozen,
    is_nondiff,
    is_tree_equal,
    tree_diagram,
    tree_indent,
    tree_mermaid,
    tree_repr,
    tree_str,
    tree_summary,
    unfreeze,
)

from . import nn
from .operators import diff, value_and_diff

__all__ = (
    # general utils
    "TreeClass",
    "is_tree_equal",
    "field",
    # pprint utils
    "tree_diagram",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_indent",
    "tree_summary",
    "tree_trace_summary",
    # freeze/unfreeze utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    # serket
    "nn",
    "diff",
    "value_and_diff",
)


__version__ = "0.2.0b6"
