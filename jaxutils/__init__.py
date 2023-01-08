# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from .pytree import PyTree
from .data import Dataset, verify_dataset
from .dict import (
    concat_dictionaries,
    merge_dictionaries,
    sort_dictionary,
    dict_array_coercion,
)
from .parameters import (
    ParameterState,
    initialise,
    recursive_items,
    recursive_complete,
)

__authors__ = "Thomas Pinder, Daniel Dodd"
__license__ = "MIT"
__emails__ = "tompinder@live.co.uk, d.dodd1@lancaster.ac.uk"
__license__ = "Apache 2.0"
__description__ = "Utilities for JAXGaussianProcesses"
__url__ = "https://github.com/JaxGaussianProcesses/JaxUtils"
__contributors__ = (
    "https://github.com//JaxGaussianProcesses/JaxUtils/graphs/contributors"
)


__all__ = [
    "PyTree",
    "Dataset",
    "verify_dataset",
    "concat_dictionaries",
    "merge_dictionaries",
    "sort_dictionary",
    "dict_array_coercion",
    "ParameterState",
    "initialise",
    "recursive_items",
    "recursive_complete",
]

from . import _version

__version__ = _version.get_versions()["version"]
