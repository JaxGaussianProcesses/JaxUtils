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
from typing import Sequence, Callable, Union
from .pytree import PyTree
from .node import node_at
from ._utils import _to_callable


class _PyTreeNodeUpdateRef:

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

    def replace(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.pytree,
            replace_fn=lambda node: node._replace(**kwargs),
        )

    def replace_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.pytree,
            replace_fn=lambda node: node._replace_meta(**kwargs),
        )

    def update_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.pytree,
            replace_fn=lambda node: node._update_meta(**kwargs),
        )


class _PyTreeNodeUpdateHelper:
    """Helper class for updating a PyTree."""

    __slots__ = ("pytree",)

    def __init__(self, pytree: PyTree) -> None:
        self.pytree = pytree

    def __getitem__(
        self,
        where: Union[Callable[[PyTree], Sequence[PyTree]], Sequence[str], Ellipsis],
    ) -> _PyTreeNodeUpdateRef:

        return _PyTreeNodeUpdateRef(self.pytree, _to_callable(where))

    def __repr__(self) -> str:
        return f"_PyTreeUpdateHelper({repr(self.pytree)})"
