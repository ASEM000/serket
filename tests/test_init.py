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


from __future__ import annotations

import pytest

from serket.nn.initialization import def_init_entry


def test_def_init_entry():
    def_init_entry("bob", lambda key, shape, dtype: None)

    with pytest.raises(ValueError):
        # duplicate entry
        def_init_entry("bob", lambda key, shape, dtype: None)

    with pytest.raises(ValueError):
        # invalid signature
        def_init_entry("one", lambda key, shape: None)
