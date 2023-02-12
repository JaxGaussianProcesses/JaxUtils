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

from __future__ import annotations

import jax.tree_util as jtu
import jax
from jax import lax

from dataclasses import fields, field
from typing import Any, Callable, NamedTuple
from collections import namedtuple

import equinox as eqx

from .bijectors import Bijector

"""NamedTuple for storing metadata (i.e., trainables and bijectors). """
_Meta = namedtuple("_Meta", ["trainables", "bijectors"])


class _cached_static_property:
    """Decorator to cache result of static immutable properties of a PyTree.

    Courtesy of Kernex library.

    !!! note
        The decorated property must *NOT* contain any dynamic attributes / PyTree leaves.
    """

    def __init__(self, static_property: Callable):
        self.name = static_property.__name__
        self.func = static_property

    def __get__(self, instance, owner):
        attr = self.func(instance)
        object.__setattr__(instance, self.name, attr)
        return attr


def param(transform: Bijector, trainable: bool = True, **kwargs: Any):
    """Used for marking default parameter transformations.

    !!! example

        ```python
        class MyModule(jaxutils.Module):
            param_a: float = jaxutils.param(jaxutils.Identity)
            param_b: float = jaxutils.param(jaxutils.Softplus)
        ```

    All PyTree leaves of the Module must be marked.

    Args:
        transform (Bijector): The default bijector that should be should upon Module initialisation.
        **kwargs (Any): If any are passed then they are passed on to `datacalss.field`.
        (Recall that JaxUtils uses dataclasses for its modules, based on Equinox's infrastructure.)
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "transform" in metadata:
        raise ValueError("Cannot use metadata with `transform` already set.")
    metadata["transform"] = transform
    if "trainable" in metadata:
        raise ValueError("Cannot use metadata with `trainable` already set.")
    metadata["trainable"] = trainable
    if "param" in metadata:
        raise ValueError("Cannot use metadata with `param` already set.")
    metadata["param"] = True

    return field(**kwargs)


def static(**kwargs: Any):
    """Alias of `equinox.static_field`. Provided for convenience.

    Used for marking that a field should _not_ be treated as a leaf of the PyTree
    of a `jaxutils.Module`/ `equinox.Module`. (And is instead treated as part of the structure, i.e.
    as extra metadata.)
    !!! example
        ```python
        class MyModule(jaxutils.Module):
            normal_field: int
            static_field: int = equinox.static_field()
        mymodule = MyModule("normal", "static")
        leaves, treedef = jtu.tree_flatten(mymodule)
        assert leaves == ["normal"]
        assert "static" in str(treedef)
        ```
    Args:
     **kwargs (Any): If any are passed then they are passed on to `datacalss.field`.
        (Recall that JaxUtils uses dataclasses for its modules, based on Equinox's infrastructure.)
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    return field(**kwargs)


def constrain(obj: Module) -> Module:
    """
    Transform model parameters to the constrained space for corresponding
    bijectors.

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.

    Returns:
        Module: PyTree tranformed to the constrained space.
    """
    return jtu.tree_map(
        lambda leaf, transform: transform.forward(leaf), obj, obj.bijectors
    )


def unconstrain(obj: Module) -> Module:
    """
    Transform model parameters to the unconstrained space for corresponding
    bijectors.

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.

    Returns:
        Module: PyTree tranformed to the unconstrained space.
    """
    return jtu.tree_map(
        lambda leaf, transform: transform.inverse(leaf), obj, obj.bijectors
    )


def stop_gradients(obj: Module) -> Module:
    """
    Stop the gradients flowing through parameters whose trainable status is
    False.

    Args:
        obj (Module): PyTree object to stop gradients for.

    Returns:
        Module: PyTree with gradients stopped.
    """

    def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
        return lax.cond(trainable, lambda x: x, lax.stop_gradient, leaf)

    return jtu.tree_map(lambda *_: _stop_grad(*_), obj, obj.trainables)


def _unpack_meta(obj: Module) -> NamedTuple:
    """Unpack metadata (i.e., trainables and bijectors) defined by the `param` field.

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.

    Returns:
        NamedTuple: A namedtuple with fields `trainables` and `bijectors`.

    Raises:
        ValueError: If a leaf is not marked as a parameter and its type is not a subclass of `Module`.
    """
    trainables, bijectors = [], []

    for field_ in fields(obj):

        if field_.metadata.get("static", False):
            continue

        if field_.metadata.get("param", False):
            trainables.append(field_.metadata["trainable"])
            bijectors.append(field_.metadata["transform"])
            continue

        if issubclass(field_.type, Module):
            for trainable, bijector in zip(*_unpack_meta(field_.type)):
                trainables.append(trainable)
                bijectors.append(bijector)
            continue

        raise ValueError(
            f"Field `{field_.name}` must either be: \n- marked as a parameter via by the `param` field\n- marked as static via the `static` field metadata\n- a jaxutils.Module."
        )

    return _Meta(trainables, bijectors)


class Module(eqx.Module):
    """Base Module object.

    This object is essentially an Equinox Module (i.e., a registered PyTree dataclass with static field markers),
    with modifications to handle bijector transformations and trainability statuses.

    !!! example
        ```python
        class MyModule(jaxutils.Module):
            param_a: float = jaxutils.param(jaxutils.Identity)
            param_b: float = jaxutils.param(jaxutils.Softplus)
        ```

    All PyTree leaves of the Module must be marked with the `param` field.
    """

    def __new__(
        cls,
        __meta__: _Meta = None,
        *args: Any,
        **kwargs: Any,
    ) -> Module:
        """This method is defined to set the `__meta__` attribute (as we are working with frozen dataclasses!).

        Args:
            __meta__ (_Meta.) Metadata that define the models' trainables and bijectors PyTree leaves.
            *args (Any). Arguments.
            **kwargs (Any). Keyword arguments.

        Returns:
            Module. An instance of the JaxUtils Module class.
        """
        obj = super().__new__(cls)
        if __meta__ is None:
            __meta__ = _unpack_meta(obj)
        object.__setattr__(obj, "__meta__", __meta__)
        return obj

    @_cached_static_property  # Caching fine here since `trainables` are static and `Module` is frozen/inmutable.
    def trainables(self) -> Module:
        """Return boolean Module comprising trainability statuses for the Module.

        Returns:
            Module: Boolean Module comprising trainability statuses for the Module.
        """
        return jtu.tree_structure(self).unflatten(self.__meta__.trainables)

    @_cached_static_property  # Caching fine here since bijectors are static and `Module` is frozen/inmutable.
    def bijectors(self) -> Module:
        """Return the Bijector Module comprising transformations for parameters to and from the constrained and unconstrained spaces.

        Returns:
            Module: The Bijector Module of parameter transformations for the Module.
        """
        return jtu.tree_structure(self).unflatten(self.__meta__.bijectors)

    def set_trainables(self, tree: Module) -> Module:
        """Set parameter trainability status for the Module.

        Args:
            tree (Module): The boolean tree of trainability status comprising the same tree structure as the underlying Module.

        Returns:
            Module: A new instance with the updated trainablility status tree.
        """

        if not isinstance(tree, Module):
            raise TypeError("Tree must be a JaxUtils Module.")

        if not jtu.tree_structure(tree) == jtu.tree_structure(self):
            raise ValueError("Tree must have the same structure as the Module.")

        return self.__update_meta__(
            trainables=jtu.tree_leaves(tree), bijectors=self.__meta__.bijectors
        )

    def set_bijectors(self, tree: Module) -> Module:
        """Set parameter transformations for the Module.

        Args:
            tree (Module): The bijector tree of parameter transformations comprising the same tree structure as the underlying Module.

        Returns:
            Module: A new instance with the updated trainablility status tree.
        """
        if not isinstance(tree, Module):
            raise TypeError("tree must be a JaxUtils Module.")

        if not jtu.tree_structure(
            jtu.tree_map(
                lambda _: True, tree, is_leaf=lambda x: isinstance(x, Bijector)
            )
        ) == jtu.tree_structure(self):
            raise ValueError(
                "bijectors tree must have the same structure as the Module."
            )

        return self.__update_meta__(
            trainables=self.__meta__.trainables, bijectors=jtu.tree_leaves(tree)
        )

    def __update_meta__(self, trainables, bijectors) -> Module:
        """Update Module meta through a new instance."""
        new = self.__class__.__new__(
            cls=self.__class__, __meta__=_Meta(trainables, bijectors)
        )

        for field_ in fields(self):
            object.__setattr__(new, field_.name, self.__dict__[field_.name])

        return new

    def tree_flatten(self):
        """Identical to that of Equinox, except for the addition of the `meta` component."""
        dynamic_field_names = []
        dynamic_field_values = []
        static_field_names = []
        static_field_values = []
        meta = [self.__meta__.trainables, self.__meta__.bijectors]

        for field_ in fields(self):
            name = field_.name
            try:
                value = self.__dict__[name]
            except KeyError:
                continue
            if field_.metadata.get("static", False):
                static_field_names.append(name)
                static_field_values.append(value)
            else:
                dynamic_field_names.append(name)
                dynamic_field_values.append(value)

        return tuple(dynamic_field_values), (
            tuple(dynamic_field_names),
            tuple(static_field_names),
            tuple(static_field_values),
            tuple(meta),
        )

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        """Identical to that of Equinox, except for the addition of the `meta` component."""

        dynamic_field_names, static_field_names, static_field_values, meta = aux

        self = cls.__new__(cls=cls, __meta__=_Meta(*meta))
        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)
        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)

        return self


__all__ = [
    "Module",
    "param",
    "static",
    "constrain",
    "unconstrain",
    "stop_gradients",
]
