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

import jax
import equinox

from jax import lax
from dataclasses import fields, field
from typing import Any, Callable, Tuple, List, Dict

from .bijectors import Bijector, Identity
from .distributions import Distribution
from ._pytree import _PyTreeUpdateHelper
from ._meta import _MetaUpdateHelper, _meta_map
from ._cached import _cached_static_property


def param(
    transform: Bijector,
    trainable: bool = True,
    prior: Distribution = None,
    **kwargs: Any,
):
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

    # TODO: Type check.

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
    if "prior" in metadata:
        raise ValueError("Cannot use metadata with `prior` already set.")
    metadata["prior"] = prior
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
    return _meta_map(
        lambda leaf, meta: meta.get("transform", Identity).forward(leaf), obj
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
    return _meta_map(
        lambda leaf, meta: meta.get("transform", Identity).inverse(leaf), obj
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
    # TODO: Remove the tree_map and work with the leaves directly.
    # This is since object metadata already comprises flattened leaves.

    def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
        """Stop gradients flowing through a given leaf if it is not trainable."""
        return lax.cond(trainable, lambda x: x, lax.stop_gradient, leaf)

    return _meta_map(
        lambda leaf, meta: _stop_grad(leaf, meta.get("trainable", True)), obj
    )


# TODO: Rewrite this as a generator.
def _default_meta(obj: Module) -> tuple[Dict]:
    meta = []

    for field_, node_ in (
        [f, getattr(obj, f.name)]
        for f in obj.__dataclass_fields__.values()
        if not f.metadata.get("static", False)
    ):

        if isinstance(node_, Module):
            for meta_ in _default_meta(node_):
                meta.append(meta_)

        elif isinstance(node_, list | tuple):
            for item in node_:
                for meta_ in _default_meta(item):
                    meta.append(meta_)
        else:
            meta.append(
                {**field_.metadata}
            )  # Have to do this otherwise can't pickle the Module.

    return meta


class Module(equinox.Module):
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
        __meta_func__: Callable[[Module], List[Dict]] = _default_meta,
        **kwargs: Any,
    ) -> Module:
        """This method is defined to set the `__meta__` attribute (as we are working with frozen dataclasses!).

        Args:
            *args (Any): Arguments.
            __meta_func__ (Callable[[Module], List[Dict]], optional): Function to generate the Module's meta. Defaults to _default_meta.
            **kwargs (Any): Keyword arguments.

        Returns:
            Module. An instance of the `jaxutils.Module`.
        """
        obj = super().__new__(cls)
        object.__setattr__(obj, "__meta_func__", __meta_func__)
        return obj

    @_cached_static_property
    def __meta__(self) -> List[Dict]:
        """Return the Module's meta.

        Returns:
            _Meta: The Module's meta.
        """
        return self.__meta_func__(self)

    def __set_meta_func__(
        self, __meta_func__: Callable[[Module], List[Dict]]
    ) -> Module:
        """Set function that generates parameter meta through a new Module instance.

        Example:
            TODO!

        Args:
            __meta_func__ (Callable[[Module], List[Dict]]): Function to generate the Module's meta.

        Returns:
            Module: A new instance of the Module with updated meta.
        """
        cls = self.__class__

        new = cls.__new__(
            cls=cls,
            __meta_func__=__meta_func__,
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
            self.__meta__,
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

        self = cls.__new__(
            cls=cls,
            __meta_func__=lambda _: meta,
        )

        for name, value in zip(dynamic_field_names, dynamic_field_values):
            object.__setattr__(self, name, value)
        for name, value in zip(static_field_names, static_field_values):
            object.__setattr__(self, name, value)

        return self

    @property
    def tree_at(self):
        return _PyTreeUpdateHelper(self)

    @property
    def meta_at(self):
        return _MetaUpdateHelper(self)

    def training_transforms(self):
        class _ConstrainHelper:

            slots = ("module",)

            def __init__(self, module):
                self.module = module

            def __enter__(self):
                return constrain(self.module)

            def __exit__(self, *args):
                return unconstrain(self.module)

        return _ConstrainHelper(self)

    def constrain(self):
        return constrain(self)

    def unconstrain(self):
        return unconstrain(self)

    def stop_gradients(self):
        class _StopGradientsHelper:

            slots = ("module",)

            def __init__(self, module):
                self.module = module

            def __enter__(self):
                return stop_gradients(self.module)

            def __exit__(self, *args):
                return True

        return _StopGradientsHelper(self)
