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
from typing import Sequence, Callable, Union, Dict
from .bijectors import Bijector
from .distributions import Distribution
from .module import Module
from .pytree import PyTree
from .node import node_at


def _replace_trainables(module: Module, **kwargs: Dict[str, bool]) -> Module:
    """Replace the trainability status of local nodes of the PyTree."""
    _meta_wrapper = {k: {"trainable": v} for k, v in kwargs.items()}
    return module._update_meta(**_meta_wrapper)


def _replace_bijectors(module: Module, **kwargs: Dict[str, Bijector]) -> Module:
    """Replace the bijectors of local nodes of the PyTree."""
    _meta_wrapper = {k: {"bijector": v} for k, v in kwargs.items()}
    return module._update_meta(**_meta_wrapper)


def _replace_priors(module: Module, **kwargs: Dict[str, Distribution]) -> Module:
    """Replace the priors of local nodes of the PyTree."""
    _meta_wrapper = {k: {"prior": v} for k, v in kwargs.items()}
    return module._update_meta(**_meta_wrapper)


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
            operator=lambda node: node._replace(**kwargs),
            node_type=Module,
        )

    def set_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            operator=lambda node: node._replace_meta(**kwargs),
            node_type=Module,
        )

    def update_meta(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            operator=lambda node: node._update_meta(**kwargs),
            node_type=Module,
        )

    def trainables(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            operator=lambda node: _replace_trainables(node, **kwargs),
            node_type=Module,
        )

    def bijectors(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            operator=lambda node: _replace_bijectors(node, **kwargs),
            node_type=Module,
        )

    def priors(self, **kwargs) -> PyTree:
        return node_at(
            where=self.where,
            pytree=self.module,
            operator=lambda node: _replace_priors(node, **kwargs),
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

        return _ModuleNodeUpdateRef(self.module, where)

    def __repr__(self) -> str:
        return f"_ModuleNodeUpdateHelper({repr(self.module)})"
