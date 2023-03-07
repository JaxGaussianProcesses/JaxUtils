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
from jax import lax
from dataclasses import field
from typing import Any, Dict
from .distributions import Distribution
from .bijectors import Bijector, Identity
from .pytree import PyTree, _metadata_map


def param(
    bijector: Bijector,
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
    if "bijector" in metadata:
        raise ValueError("Cannot use metadata with `bijector` already set.")
    metadata["bijector"] = bijector
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


def _constrain(obj: Module) -> Module:
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
    return _metadata_map(
        lambda leaf, meta: meta.get("bijector", Identity).forward(leaf), obj
    )


def _unconstrain(obj: Module) -> Module:
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
    return _metadata_map(
        lambda leaf, meta: meta.get("bijector", Identity).inverse(leaf), obj
    )


def _stop_gradients(obj: Module) -> Module:
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

    return _metadata_map(
        lambda leaf, meta: _stop_grad(leaf, meta.get("trainable", True)), obj
    )


class Module(PyTree):
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

    def _replace_trainables(self, **kwargs: Dict[str, bool]) -> Module:
        _meta_wrapper = {k: {"trainable": v} for k, v in kwargs.items()}
        return self._update_meta(**_meta_wrapper)

    def _replace_bijectors(self, **kwargs: Dict[str, Bijector]) -> Module:
        _meta_wrapper = {k: {"bijector": v} for k, v in kwargs.items()}
        return self._update_meta(**_meta_wrapper)

    def _replace_priors(self, **kwargs: Dict[str, Distribution]) -> Module:
        _meta_wrapper = {k: {"prior": v} for k, v in kwargs.items()}
        return self._update_meta(**_meta_wrapper)

    @property
    def at(self):
        from ._module import _ModuleNodeUpdateHelper

        return _ModuleNodeUpdateHelper(self)

    @property
    def module_at(self):
        from ._module import _ModuleNodeUpdateHelper

        return _ModuleNodeUpdateHelper(self)

    def training_transforms(self):
        class _ConstrainHelper:

            slots = ("module",)

            def __init__(self, module):
                self.module = module

            def __enter__(self):
                return self.module.constrain()

            def __exit__(self, *args):
                return True

        return _ConstrainHelper(self)

    def constrain(self):
        return _constrain(self)

    def unconstrain(self):
        return _unconstrain(self)

    def stop_gradients(self):
        class _StopGradientsHelper:

            slots = ("module",)

            def __init__(self, module):
                self.module = module

            def __enter__(self):
                return _stop_gradients(self.module)

            def __exit__(self, *args):
                return True

        return _StopGradientsHelper(self)
