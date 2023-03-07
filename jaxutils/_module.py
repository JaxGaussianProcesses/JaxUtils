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
from typing import Sequence, Callable, Union, Iterable
from .module import Module
from .pytree import PyTree
from .node import node_at


def _to_callable(
    where: Union[Callable[[Module], Sequence[Module]], Sequence[str], Ellipsis]
) -> Callable[[Module], Sequence[Module]]:

    if isinstance(where, str):
        where = eval("lambda x: x." + where)

    if isinstance(where, Iterable):

        def _to_path(it: Iterable):
            return "".join(
                [str(elem) if not isinstance(elem, str) else "." + elem for elem in it]
            )

        where = eval("lambda x: x" + _to_path(where))

    if isinstance(where, type(Ellipsis)):
        where = lambda x: x

    return where


class _ModuleNodeUpdateRef:

    __slots__ = (
        "module",
        "where",
    )

    def __init__(self, module: Module, where: Union[Callable, Sequence[str]]) -> None:
        self.module = module
        self.where = where

    def __repr__(self) -> str:
        return f"_ModuleNodeUpdateRef({repr(self.module)}, {repr(self.where)})"

    def get(self) -> PyTree:
        return self.where(self.pytree)

    def set(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._replace(**kwargs),
            node_type=Module,
        )

    def set_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._replace_meta(**kwargs),
            node_type=Module,
        )

    def update_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._update_meta(**kwargs),
            node_type=Module,
        )

    def trainables(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._replace_trainables(**kwargs),
            node_type=Module,
        )

    def bijectors(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._replace_bijectors(**kwargs),
            node_type=Module,
        )

    def priors(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            replace_fn=lambda node: node._replace_priors(**kwargs),
            node_type=Module,
        )


class _ModuleNodeUpdateHelper:
    """Helper class for updating a PyTree."""

    __slots__ = ("module",)

    def __init__(self, module: Module) -> None:
        self.module = module

    def __getitem__(
        self,
        where: Union[Callable[[Module], Sequence[Module]], Sequence[str], Ellipsis],
    ) -> _ModuleNodeUpdateRef:

        where = _to_callable(where)

        return _ModuleNodeUpdateRef(self.module, where)

    def __repr__(self) -> str:
        return f"_ModuleNodeUpdateHelper({repr(self.module)})"
