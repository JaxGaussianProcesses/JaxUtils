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
from typing import Any, Callable, NamedTuple, Tuple
from collections import namedtuple

import equinox as eqx

from .bijectors import Bijector

"""NamedTuple for storing transformations and trainability status metadata of `jaxutils.Module` PyTree leaves.

Args:
    trainables (Tuple[bool]): Whether the PyTree leaf is trainable.
    bijectors (Tuple[Bijector]): The bijector that should be applied to the PyTree leaf.

Example:
    This example shows us creating a module with two parameters, and accessing the metadata of the module. 
    This is part of the non-public API, and is not intended to be used directly by users.

    >>> import jaxutils as ju
    >>> import jax.numpy as jnp
    >>>
    >>> # Create a bijector for demonstration purposes.
    >>> class Test(ju.Bijector):
    >>>     def __init__(self):
    >>>         self.forward = jnp.tanh
    >>>         self.inverse = jnp.arctanh
    >>> 
    >>> # Create a module with two parameters.
    >>> class MyModule(ju.Module):
    >>>     param_a: float = ju.param(Test())
    >>>     param_b: float = ju.param(Test())
    >>>
    >>>     def __call__(self, x):
    >>>         return self.param_a * x + self.param_b
    >>>
    >>> module = MyModule(param_a=1.0, param_b=1.0)
    >>>
    >>> # The `__meta__` attribute is a `_Meta` object.
    >>> print(module.__meta__)
    _Meta(trainables=(True, True), bijectors=(Test(forward=<wrapped function <lambda>>, inverse=<wrapped function <lambda>>), Test(forward=<wrapped function <lambda>>, inverse=<wrapped function <lambda>>)))

"""
_Meta = namedtuple("_Meta", ["trainables", "bijectors"])


class _cached_static_property:
    """Decorator to cache result of static immutable properties of a PyTree.

    Example:

        This example shows us caching the result of sqauring a static float attribute of a Module.

        >>> import jaxutils as ju
        >>>
        >>> class MyModule(ju.Module):
        >>>     static_attribute: float = ju.static()

        >>>     @_cached_static_property
        >>>     def static_property(self):
        >>>         return self.static_attribute ** 2

    Note:
        The decorated property must *NOT* contain any dynamic attributes / PyTree leaves,
        i.e., any attributes referenced in the property must be marked as static.

        For example, the following will break durin tracing since `self.dynamic_attribute` is not static:

        >>> import jaxutils as ju
        >>>
        >>> class MyModule(ju.Module):
        >>>     static_attribute: float = ju.static()
        >>>     dynamic_attribute: float = ju.param(ju.Identity)
        >>>
        >>>     @_cached_static_property
        >>>     def static_property(self):
        >>>         return self.static_attribute ** 2 + self.dynamic_attribute
    """

    def __init__(self, static_property: Callable):
        """Here we store the name of the property and the function itself."""
        self.name = static_property.__name__
        self.func = static_property

    def __get__(self, instance, owner):
        """Here we cache the result of the property function, by overwriting the attribute with the result."""
        attr = self.func(instance)
        object.__setattr__(instance, self.name, attr)
        return attr


def param(transform: Bijector, trainable: bool = True, **kwargs: Any):
    """Used for marking default parameter transformations.

    All PyTree leaves of the `jaxutils.Module` must be marked by this function, `jaxutils.static`, or must be of type `jaxutils.Module`.

    Example:
        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b
        >>>
        >>> module = MyModule(param_a=1.0, param_b=1.0)

        We can access the trainability status of the parameters using the `trainables` property.

        >>> # Print the trainability status PyTree
        >>> print(module.trainables)
        LinearModel(weight=True, bias=True)

        And we can access the bijectors of the parameters using the `bijectors` property.

        >>> # Print the bijectors of the PyTree
        >>> print(module.bijectors)
        LinearModel(
            weight=Bijector(forward=<function <lambda>>, inverse=<function <lambda>>),
            bias=Bijector(forward=<function <lambda>>, inverse=<function <lambda>>)
            )

        Under the hood, the `param` function is used to create a `dataclasses.field` object with the `transform` and `trainable` metadata set,
        which is then used to initialise the non-public and static `Module.__meta__` attribute.

        >>> # Print the trainability status leaves from the `__meta__` attribute.
        >>> print(module.__meta__.trainables)
        (True, False)
        >>>
        >>> # Print the bijectors of the leaves from the `__meta__` attribute.
        >>> print(module.__meta__.bijectors)
        (Identity(forward=<wrapped function <lambda>>, inverse=<wrapped function <lambda>>), Softplus(forward=<wrapped function <lambda>>, inverse=<wrapped function <lambda>>))

    Args:
        transform (Bijector): The default bijector that should be should upon Module initialisation.
        **kwargs (Any): If any are passed then they are passed on to `datacalss.field`.
        (Recall that JaxUtils uses dataclasses for its modules, based on Equinox's infrastructure.)

    Returns:
        A `dataclasses.field` object with the `transform` and `trainable` metadata set.
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

    Example:
        This example shows us creating a module with a static field, and then flattening the module.

        >>> import jaxutils as ju
        >>>
        >>> class MyModule(ju.Module):
        >>>     normal_field: int
        >>>     static_field: int = ju.static()
        >>>
        >>> mymodule = MyModule("normal", "static")
        >>> leaves, treedef = jax.tree_flatten(mymodule)
        >>> print(leaves)
        ['normal']
        >>> print(treedef)
        PyTreeDef(<class 'jaxutils.module.MyModule'>, {'static_field': *})
        >>>
        >>> # The same example using `equinox.static_field`
        >>> import equinox as eq
        >>>
        >>> class MyModule(ju.Module):
        >>>     normal_field: int
        >>>     static_field: int = eq.static_field()
        >>>
        >>> mymodule = MyModule("normal", "static")
        >>> leaves, treedef = jax.tree_flatten(mymodule)
        >>> print(leaves)
        ['normal']
        >>> print(treedef)
        PyTreeDef(<class 'jaxutils.module.MyModule'>, {'static_field': *})

    Args:
        **kwargs (Any): If any are passed then they are passed on to `dataclass.field`.
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
    """Transform model parameters to the constrained space according to their defined bijectors.

    Example:
        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b
        >>>
        >>> module = MyModule(param_a=1.0, param_b=1.0)

        We can constrain the parameters using the `constrain` function.

        >>> constrained_module = ju.constrain(module)
        >>> print(constrained_module)
        MyModule(param_a=1.0, param_b=0.0)

        And we can unconstrain the parameters using the `unconstrain` function.

        >>> unconstrained_module = ju.unconstrain(constrained_module)
        >>> print(unconstrained_module)
        MyModule(param_a=1.0, param_b=1.0)

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.

    Returns:
        Module: PyTree tranformed to the constrained space.
    """
    return jtu.tree_map(
        lambda leaf, transform: transform.forward(leaf), obj, obj.bijectors
    )


def unconstrain(obj: Module) -> Module:
    """Transform model parameters to the unconstrained space according to their defined bijectors.

    Example:
        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b
        >>>
        >>> module = MyModule(param_a=1.0, param_b=1.0)

        We can constrain the parameters using the `constrain` function.

        >>> constrained_module = ju.constrain(module)
        >>> print(constrained_module)
        MyModule(param_a=1.0, param_b=0.0)

        And we can unconstrain the parameters using the `unconstrain` function.

        >>> unconstrained_module = ju.unconstrain(constrained_module)
        >>> print(unconstrained_module)
        MyModule(param_a=1.0, param_b=1.0)

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
    False. This is useful for stopping parameters from being updated during training.

    Example:
        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b

        We now create a dummy objective function and check the gradients of the parameters.

        >>> module = MyModule(param_a=5.0, param_b=7.0)
        >>> def dummy_objective(module, x):
        ...     module = ju.stop_gradients(module) # Stop gradients flowing through `param_b`
        ...     return jnp.sum(module(x))
        >>> g = jax.grad(dummy_objective)(module, 1.0)

        We can see that the gradient of `param_a` is 1.0, but the gradient of `param_b` is 0.0, as expected.

        >>> print(g.param_a, g.param_b)
        1.0 0.0

    Args:
        obj (Module): PyTree object to stop gradients for.

    Returns:
        Module: PyTree with gradients stopped.
    """

    def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
        """Stop gradients flowing through a given leaf if it is not trainable."""
        return lax.cond(trainable, lambda x: x, lax.stop_gradient, leaf)

    return jtu.tree_map(lambda *_: _stop_grad(*_), obj, obj.trainables)


def _unpack_meta(obj: Module) -> NamedTuple:
    """Unpack trainable and tranformation metadata defined by the `jaxutils.param` field of a `jaxutils.Module`.

    This is used to build the leaves of the `trainables` and `bijectors` attributes defined in the `__meta__` attribute of a `jaxutils.Module`, during class instantiation.

    Example:
        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b
        >>>
        >>> module = MyModule(param_a=1.0, param_b=1.0)

        We unpack the metadata from the `param` fields using the `_unpack_meta` function.

        >>> meta_named_tuple = _unpack_meta(module)
        >>> print(meta_named_tuple)
        _Meta(trainables=(True, False), bijectors=(Bijector(forward=<function <lambda>>, inverse=<function <lambda>>), Bijector(forward=<function <lambda>>, inverse=<function <lambda>>)))

        We can see this is equivalent to the `__meta__` attribute of the module.
        >>> assert meta_named_tuple == module.__meta__
        True

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

    return _Meta(tuple(trainables), tuple(bijectors))


class Module(eqx.Module):
    """Base Module object.

    This object is essentially an Equinox Module (i.e., a registered PyTree dataclass with static field markers),
    with modifications to handle bijector transformations and trainability statuses.

    Example:

        TODO: More thorough example.

        This example shows us creating a module with two parameters, with differing transformations and trainability statuses.

        >>> import jaxutils as ju
        >>> import jax.numpy as jnp
        >>>
        >>> class MyModule(ju.Module):
        >>>     param_a: float = ju.param(ju.Identity, trainable=True)
        >>>     param_b: float = ju.param(ju.Softplus, trainable=False)
        >>>
        >>>     def __call__(self, x):
        >>>         return self.param_a * x + self.param_b

    Note:
        All attributes of the Module must be marked with either a `param` or `static` field or the attributes type needs to be subclass of `jaxutils.Module`.
    """

    def __new__(
        cls,
        *args: Any,
        __meta__: _Meta = None,
        **kwargs: Any,
    ) -> Module:
        """This method is defined to set the `__meta__` attribute (as we are working with frozen dataclasses!).

        Args:
            __meta__ (_Meta): Metadata that defines the Module's trainables and bijectors PyTree leaves.
            *args (Any): Arguments.
            **kwargs (Any): Keyword arguments.

        Returns:
            Module. An instance of the `jaxutils.Module`.
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

        Example:
            TODO!

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

        Example:
            TODO!

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

    def __update_meta__(self, trainables: Tuple, bijectors: Tuple) -> Module:
        """Update Module meta through a new instance.

        Example:
            TODO!

        Args:
            trainables (Tuple): The new leaves of the trainables PyTree.
            bijectors (Tuple): The new leaves of the bijectors PyTree.

        Returns:
            Module: A new instance of the Module with updated meta.
        """
        new = self.__class__.__new__(
            cls=self.__class__, __meta__=_Meta(trainables, bijectors)
        )

        for field_ in fields(self):
            object.__setattr__(new, field_.name, self.__dict__[field_.name])

        return new

    def tree_flatten(self) -> Tuple[Tuple, Tuple]:
        """Identical to that of Equinox, except for the addition of a meta component.

        Returns:
            Tuple: A tuple of the Module's dynamic and static fields.
        """
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
    def tree_unflatten(cls, aux: Tuple, dynamic_field_values: Tuple) -> Module:
        """Identical to that of Equinox, except for the addition of a meta component.

        Args:
            aux (Tuple): Auxiliary data.
            dynamic_field_values (Tuple): Dynamic field values.

        Returns:
            Module: An instance of the `jaxutils.Module` class.
        """
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
