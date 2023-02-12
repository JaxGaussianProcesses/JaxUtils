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
from typing import Any, Callable, List, Tuple

import equinox as eqx

from .bijectors import Bijector


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
        __trainables_leaves__: List[bool] = None,
        __bijectors_leaves__: List[Bijector] = None,
        __unpack_meta__: bool = True,
        *args: Any,
        **kwargs: Any,
    ) -> Module:
        """This is used to set the trainables and bijectors functions. As we are working with frozen dataclasses.

        Args:
            __trainables_func__ (Callable[[Module], Module]). The function that constructs the trainables PyTree from `self`.
            __bijectors_func__ (Callable[[Module], Module]). The function that constructs the bijectors PyTree from `self`.
            __unpack_meta__ (bool). Whether to unpack the metadata from the `param` field.
            *args (Any). Arguments.
            **kwargs (Any). Keyword arguments.

        Returns:
            Module. An instance of the JaxUtils Module class.
        """

        obj = super().__new__(cls)

        def _unpack_meta(obj: Module) -> Tuple[List[bool], List[Bijector]]:
            """Unpack trainables from metadata defined by the `param` field.

            If a leaf is not marked as a parameter then this will raise an error.
            """
            train_meta_ = []
            bij_meta_ = []

            for field_ in fields(obj):

                if field_.metadata.get("static", False):
                    continue

                if field_.metadata.get("param", False):
                    train_meta_.append(field_.metadata["trainable"])
                    bij_meta_.append(field_.metadata["transform"])
                    continue

                if issubclass(field_.type, Module):
                    for trainable_, bijector_ in zip(*_unpack_meta(field_.type)):
                        train_meta_.append(trainable_)
                        bij_meta_.append(bijector_)
                    continue

                raise ValueError(
                    f"Field `{field_.name}` must either be: \n- marked as a parameter via by the `param` field\n- marked as static via the `static` field metadata\n- a jaxutils.Module."
                )

            return train_meta_, bij_meta_

        if __unpack_meta__:
            train_meta_, bij_meta_ = _unpack_meta(obj)

        if __trainables_leaves__ is None:
            __trainables_leaves__ = train_meta_

        if __bijectors_leaves__ is None:
            __bijectors_leaves__ = bij_meta_

        object.__setattr__(obj, "__trainables_leaves__", __trainables_leaves__)
        object.__setattr__(obj, "__bijectors_leaves__", __bijectors_leaves__)

        return obj

    @_cached_static_property  # Cacheing is fine here as the trainables are static and `Module` is frozen/inmutable.
    def trainables(self) -> Module:
        """Return the boolean Module comprising trainability statuses.

        Returns:
            Module: The boolean Module comprising trainability statuses for the Module.
        """
        return jtu.tree_structure(self).unflatten(self.__trainables_leaves__)

    @_cached_static_property  # Cacheing is fine here as the trainables are static and `Module` is frozen/inmutable.
    def bijectors(self) -> Module:
        """Return the Bijector Module comprising transformations for parameters to and from the constrained and unconstrained spaces.

        Returns:
            Module: The Bijector Module of parameter transformations.
        """
        return jtu.tree_structure(self).unflatten(self.__bijectors_leaves__)

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

        return self.__set_trainables_leaves__(jtu.tree_leaves(tree))

    def set_bijectors(self, tree: Module) -> Module:
        """Set parameter transformations for the Module.

        Args:
            tree (Module): The Bijector tree of parameter transformations comprising the same tree structure as the underlying Module.

        Returns:
            Module: A new instance with the updated trainablility status tree.
        """

        if not isinstance(tree, Module):
            raise TypeError("tree must be a JaxUtils Module.")

        def _is_bij(x):
            return isinstance(x, Bijector)

        if not jtu.tree_structure(
            jtu.tree_map(lambda _: True, tree, is_leaf=_is_bij)
        ) == jtu.tree_structure(self):
            raise ValueError(
                "bijectors tree must have the same structure as the Module."
            )

        return self.__set_bijectors_leaves__(jtu.tree_leaves(tree))

    def __set_trainables_leaves__(self, __trainables_leaves__: List[bool]) -> Module:
        """Set the trainables function for the class."""

        # Create new class instance, with the adjusted trainable function.
        new_instance = self.__class__.__new__(
            cls=self.__class__,
            __trainables_leaves__=__trainables_leaves__,
            __bijectors_leaves__=self.__bijectors_leaves__,
            __unpack_meta__=False,
        )

        # TODO: Might have to filter attribute dict here?
        for field_ in fields(self):
            object.__setattr__(new_instance, field_.name, self.__dict__[field_.name])

        return new_instance

    def __set_bijectors_leaves__(
        self, __bijectors_leaves__: Callable[[Module], Module]
    ) -> Module:
        """Set the bijectors function for the class."""

        # Create new class instance, with the adjusted trainable function.
        new_instance = self.__class__.__new__(
            self.__class__,
            __trainables_leaves__=self.__trainables_leaves__,
            __bijectors_leaves__=__bijectors_leaves__,
            __unpack_meta__=False,
        )

        # TODO: Might have to filter attribute dict here?
        for field_ in fields(self):
            object.__setattr__(new_instance, field_.name, self.__dict__[field_.name])

        return new_instance

    def tree_flatten(self):
        """Same as Equinox, except for the addition of the `static_meta`."""
        dynamic_field_names = []
        dynamic_field_values = []
        static_field_names = []
        static_field_values = []

        static_meta = [
            self.__trainables_leaves__,
            self.__bijectors_leaves__,
        ]

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
            tuple(static_meta),
        )

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        """Same as Equinox, except for the addition of the `static_meta`."""

        dynamic_field_names, static_field_names, static_field_values, static_meta = aux

        # These are the static functions that determine the trainable and bijector PyTree's from self.
        __trainables_leaves__, __bijectors_leaves__ = static_meta

        self = cls.__new__(
            cls=cls,
            __trainables_leaves__=__trainables_leaves__,
            __bijectors_leaves__=__bijectors_leaves__,
        )

        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)

        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)

        return self


__all__ = [
    "Module",
    "param",
    "constrain",
    "unconstrain",
    "stop_gradients",
]
