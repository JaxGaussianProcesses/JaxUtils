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
from typing import List

import equinox as eqx

from .bijectors import Bijector


class Module(eqx.Module):
    """Base class for all objects of the JaxGaussianProcesses ecosystem."""

    #TODO: Check all (dynamic) leaf nodes are marked by the `param` field.


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


def constrain(obj: Module, bij: Module) -> Module:
    """
    Transform model parameters to the constrained space for corresponding
    bijectors.

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.
        bij (Module): The PyTree object whoose leaves are the corresponding bijector transformations.

    Returns:
        Module: PyTree tranformed to the constrained space.
    """
    return jtu.tree_map(lambda leaf, transform: transform.forward(leaf), obj, bij)


def unconstrain(obj: Module, bij: Module) -> Module:
    """
    Transform model parameters to the unconstrained space for corresponding
    bijectors.

    Args:
        obj (Module): The PyTree object whoose leaves are to be transformed.
        bij (Module): The PyTree object whoose leaves are the corresponding bijector transformations.

    Returns:
        Module: PyTree tranformed to the unconstrained space.
    """
    return jtu.tree_map(lambda leaf, transform: transform.inverse(leaf), obj, bij)


def stop_gradients(obj: Module, trn: Module) -> Module:
    """
    Stop the gradients flowing through parameters whose trainable status is
    False.
    Args:
        obj (Module): PyTree object to stop gradients for.
        trn (Module): PyTree of booleans indicating whether to stop gradients for each leaf node.
    Returns:
        Module: PyTree with gradients stopped.
    """

    def _stop_grad(leaf: jax.Array, trainable: bool) -> jax.Array:
        return lax.cond(trainable, lambda x: x, lax.stop_gradient, leaf)

    return jtu.tree_map(lambda *_: _stop_grad(*_), obj, trn)


__all__ = [
    "Module",
    "param",
    "constrain",
    "unconstrain",
    "stop_gradients",
]
