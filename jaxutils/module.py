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
import jax.tree_util as jtu
from jax import lax
from dataclasses import field, Field
from typing import Any
from .distributions import Distribution
from .bijectors import Bijector, Identity
from .pytree import PyTree, _unpack_metatree_leaves


def param(
    bijector: Bijector,
    trainable: bool = True,
    prior: Distribution = None,
    **kwargs: Any,
) -> Field:
    """Used for marking default parameter transformations, trainable statuses and prior distributions for Module.

    Args:
        transform (Bijector): The default bijector that should be should upon Module initialisation.
        trainable (bool): Whether the parameter should be trainable or not.
        prior (Distribution): The prior distribution for the parameter.
        **kwargs (Any): If any are passed then they are passed on to `datacalss.field`.
        (Recall that JaxUtils uses dataclasses for its modules, based on Equinox's infrastructure.)

    Returns:
        Field: A `dataclasses.Field` object with the `bijector`, `trainable` and `prior` metadata set.
    """

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


class Module(PyTree):
    """Base Module object.

    Example:
        TODO: Give example.
    """

    @property
    def module_at(self):
        from ._module import _ModuleNodeUpdateHelper

        return _ModuleNodeUpdateHelper(self)

    def constrain(self):
        """Transform model parameters to the constrained space according to their defined bijectors.

        Returns:
            Module: tranformed to the constrained space.
        """
        return _metadata_map(
            lambda leaf, meta: meta.get("bijector", Identity).forward(leaf), self
        )

    def unconstrain(self):
        """Transform model parameters to the unconstrained space according to their defined bijectors.

        Returns:
            Module: tranformed to the unconstrained space.
        """
        return _metadata_map(
            lambda leaf, meta: meta.get("bijector", Identity).inverse(leaf), self
        )

    def stop_gradients(self):
        """Stop gradients flowing through the module.

        Returns:
            Module: with gradients stopped.
        """
        # Stop gradients flowing through a given leaf if it is not trainable.
        def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
            return lax.cond(trainable, lambda x: x, lax.stop_gradient, leaf)

        return _metadata_map(
            lambda leaf, meta: _stop_grad(leaf, meta.get("trainable", True)), self
        )


def _metadata_map(f: Callable[[Any, Dict[str, Any]], Any], pytree: PyTree) -> PyTree:
    """Apply a function to a pytree where the first argument are the pytree leaves, and the second argument are the pytree metadata leaves.

    Args:
        f (Callable[[Any, Dict[str, Any]], Any]): The function to apply to the pytree.
        pytree (PyTree): The pytree to apply the function to.

    Returns:
        PyTree: The transformed pytree.
    """
    leaves, treedef = jtu.tree_flatten(pytree)
    meta_leaves = _unpack_metatree_leaves(pytree)
    all_leaves = [leaves] + [meta_leaves]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))
