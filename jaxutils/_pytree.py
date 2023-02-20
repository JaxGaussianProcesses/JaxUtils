# Copyright 2023 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

"""This non-public module defines PyTree node updating functionality for the `jaxutils.Module` via the `jax.numpy` e.g. `.at` and `.set` syntax."""

from __future__ import annotations
from typing import Any, Sequence, Callable, Union, Iterable
from jaxtyping import PyTree
from equinox import tree_at


class _PyTreeUpdateRef:

    __slots__ = (
        "pytree",
        "where",
    )

    def __init__(self, pytree: PyTree, where: Union[Callable, Sequence[str]]) -> None:
        self.pytree = pytree
        self.where = where

    def __repr__(self) -> str:
        return f"_PyTreeUpdateRef({repr(self.pytree)}, {repr(self.where)})"

    def get(self) -> PyTree:
        return self.where(self.pytree)

    def set(self, values: Any) -> PyTree:
        return tree_at(where=self.where, pytree=self.pytree, replace=values)

    def apply(self, func: Callable) -> PyTree:
        return tree_at(where=self.where, pytree=self.pytree, replace_fn=func)


class _PyTreeUpdateHelper:
    """Helper class for updating a PyTree."""

    __slots__ = ("pytree",)

    def __init__(self, pytree: PyTree) -> None:
        self.pytree = pytree

    def __getitem__(
        self, where: Union[Callable, Sequence[str], Ellipsis]
    ) -> _PyTreeUpdateRef:

        if isinstance(where, str):
            where = eval("lambda x: x." + where)

        if isinstance(where, Iterable):

            def _to_path(it: Iterable):
                return "".join(
                    [
                        str(elem) if not isinstance(elem, str) else "." + elem
                        for elem in it
                    ]
                )

            where = eval("lambda x: x" + _to_path(where))

        if isinstance(where, type(Ellipsis)):
            where = lambda x: x

        return _PyTreeUpdateRef(self.pytree, where)

    def __repr__(self) -> str:
        return f"_PyTreeUpdateHelper({repr(self.pytree)})"
