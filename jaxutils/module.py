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
from typing import Any, Callable, List

import equinox as eqx

from .bijectors import Bijector


def default_trainables(obj: Module) -> Module:
    """
    Construct trainable statuses for each parameter. By default,
    every parameter within the model is trainable.

    Returns:
        Module: A Module of boolean trainability statuses.
    """
    return jtu.tree_map(lambda _: True, obj)


def default_bijectors(obj: Module) -> Module:
    """Given a Module object, return an equinox Module of bijectors comprising the same structure.

    Args:
        obj (Module): The PyTree object whoose default bijectors (from the param field) you would to obtain.

    Returns:
        Module: A PyTree of bijectors comprising the same structure as `obj`.
    """

    _, treedef = jtu.tree_flatten(obj)

    def _unpack_bijectors_from_meta(obj: Module) -> List[Bijector]:
        """Unpack bijectors from metatdata."""
        bijectors_ = []

        for field_ in fields(obj):
            try:
                value_ = obj.__dict__[field_.name]
            except KeyError:
                continue

            if not field_.metadata.get("static", False):

                if field_.metadata.get("transform", None) is not None:
                    bijectors_.append(field_.metadata["transform"])

                elif isinstance(value_, Module):
                    for value__ in _unpack_bijectors_from_meta(value_):
                        bijectors_.append(value__)

                else:
                    bijectors_.append(value_)

        return bijectors_

    bijectors_ = _unpack_bijectors_from_meta(obj)

    return treedef.unflatten(bijectors_)


def param(transform: Bijector):
    """Set leaf node metadata for a parameter.

    Args:
        transform (Bijector): A default bijector transformation for the parameter.

    Returns:
        A field with the metadata set.
    """
    return field(metadata={"transform": transform})


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


# TODO: Leaf node typing. E.g., to distiguish between boolean PyTree and one that is jax.Arrays.
# TODO: Leaf node checking.


class Module(eqx.Module):
    """Base Module object.

    This object is essentially an Equinox Module (i.e., a registered PyTree dataclass with static field markers),
    with modifications to handle bijector transformations and trainability statuses.

    """

    def __new__(
        cls,
        trainables_func: Callable[[Module], Module] = default_trainables,
        bijectors_func: Callable[[Module], Module] = default_bijectors,
        *args: Any,
        **kwargs: Any,
    ) -> Module:
        """This is used to set the trainables and bijectors functions. As we are working with frozen dataclasses.

        Args:
            trainables_func (Callable[[Module], Module]). The function that constructs the trainables PyTree from `self`.
            bijectors_func (Callable[[Module], Module]). The function that constructs the bijectors PyTree from `self`.
            *args (Any). Arguments.
            **kwargs (Any). Keyword arguments.

        Returns:
            Module. An instance of the JaxUtils Module class.
        """

        instance = super().__new__(cls)
        object.__setattr__(instance, "__trainables_func__", trainables_func)
        object.__setattr__(instance, "__bijectors_func__", bijectors_func)

        return instance

    @property
    def tree_def(self):
        """Return the tree definition."""
        _, tree_def_ = jtu.tree_flatten(self)
        return tree_def_

    @property
    def leaves(self):
        """Return pytree tree leaves."""
        leaves_, _ = jtu.tree_flatten(self)
        return leaves_

    @property
    def trainables(self) -> Module:
        """Return the boolean Module comprising trainability statuses.

        Returns:
            Module. The boolean Module comprising trainability statuses for the Module.
        """
        return self.__trainables_func__(self)

    @property
    def bijectors(self) -> Module:
        """Return the Bijector Module comprising transformations for parameters to and from the constrained and unconstrained spaces.

        Returns:
            Module. The Bijector Module of parameter transformations.
        """
        return self.__bijectors_func__(self)

    def set_trainables(self, tree: Module) -> Module:
        """Set parameter trainability status for the Module.

        Args:
            tree (Module). The boolean tree of trainability status comprising the same tree structure as the underlying Module.

        Returns:
            Module. A new instance with the updated trainablility status tree.
        """
        return self.__set_trainables_func__(
            lambda obj: obj.tree_def.unflatten(tree.leaves)
        )

    def set_bijectors(self, tree: Module) -> Module:
        """Set parameter transformations for the Module.

        Args:
            tree (Module). The Bijector tree of parameter transformations comprising the same tree structure as the underlying Module.

        Returns:
            Module. A new instance with the updated trainablility status tree.
        """
        return self.__set_bijectors_func__(
            lambda obj: obj.tree_def.unflatten(tree.leaves)
        )

    def __set_trainables_func__(self, func: Callable[[Module], Module]) -> Module:
        """Set the trainables function for the class."""

        # Create new class instance, with the adjusted trainable function.
        new_instance = self.__class__.__new__(
            self.__class__,
            trainables_func=func,
            bijectors_func=self.__bijectors_func__,
        )

        # TODO: Might have to filter attribute dict here?
        for field_ in fields(self):
            object.__setattr__(new_instance, field_.name, self.__dict__[field_.name])

        return new_instance

    def __set_bijectors_func__(self, func: Callable[[Module], Module]) -> Module:
        """Set the bijectors function for the class."""

        # Create new class instance, with the adjusted trainable function.
        new_instance = self.__class__.__new__(
            self.__class__,
            trainables_func=self.__trainables_func__,
            bijectors_func=func,
        )

        # TODO: Might have to filter attribute dict here?
        for field_ in fields(self):
            object.__setattr__(new_instance, field_.name, self.__dict__[field_.name])

        return new_instance

    def tree_flatten(self):
        """Same as Equinox, except for the addition of the `static_funcs`."""
        dynamic_field_names = []
        dynamic_field_values = []
        static_field_names = []
        static_field_values = []

        static_funcs = [
            self.__trainables_func__,
            self.__bijectors_func__,
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
            tuple(static_funcs),
        )

    @classmethod
    def tree_unflatten(cls, aux, dynamic_field_values):
        """Same as Equinox, except for the addition of the `static_funcs`."""

        dynamic_field_names, static_field_names, static_field_values, static_funcs = aux

        # These are the static functions that determine the trainable and bijector PyTree's from self.
        trainables_func, bijectors_func = static_funcs

        self = cls.__new__(
            cls,
            trainables_func=trainables_func,
            bijectors_func=bijectors_func,
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
