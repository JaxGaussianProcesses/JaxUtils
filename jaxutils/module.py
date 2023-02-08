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
from jax import lax

from dataclasses import fields, field
from typing import List

import equinox as eqx

from .bijectors import Bijector


class Module(eqx.Module):
    """Base class for all objects of the JaxGaussianProcesses ecosystem.""" 
   
    @property
    def trainables(self):
        return default_trainables(self)

    @property
    def bijectors(self):
        return default_bijectors(self)



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
    
    Returns:
        Module: A Module of bijectors.
    """

    _, treedef = jtu.tree_flatten(obj)

    def _unpack_bijectors_from_meta(cls: Module) -> List[Bijector]:
        """Unpack bijectors from metatdata."""
        bijectors = []

        for field_ in fields(cls):
            name = field_.name
            
            try:
                value = cls.__dict__[name]
            except KeyError:
                continue

            if not field_.metadata.get("static", False):

                if field_.metadata.get("transform", None) is not None:
                    trans = field_.metadata["transform"]
                    bijectors.append(trans)

                elif isinstance(value, Module):
                    for value_ in _unpack_bijectors_from_meta(value):
                        bijectors.append(value_)

                else:
                    bijectors.append(value)
            
        return bijectors
    
    return treedef.unflatten(_unpack_bijectors_from_meta(obj))



def param(transform: Bijector):
    """Set leaf node metadata for a parameter.

    Args:
        transform: A bijector to apply to the parameter.

    Returns:
        A field with the metadata set.
    """
    return field(metadata={"transform": transform})


def constrain(obj: Module) -> Module:
    """
    Transform model parameters to the constrained space for corresponding
    bijectors.

    Args:
        obj (Module): The Base that is to be transformed.

    Returns:
        Base: A transformed parameter set. The dictionary is equal in
            structure to the input params dictionary.
    """
    return jtu.tree_map(lambda p, t: t.forward(p), obj, obj.bijectors)


def unconstrain(obj: Module) -> Module:
    """
    Transform model parameters to the unconstrained space for corresponding
    bijectors.

    Args:
        obj (Module): The Base that is to be transformed.

    Returns:
        Base: A transformed parameter set. The dictionary is equal in
            structure to the input params dictionary.
    """
    return jtu.tree_map(lambda p, t: t.inverse(p), obj, obj.bijectors)


def stop_gradients(obj: Module) -> Module:
    """
    Stop the gradients flowing through parameters whose trainable status is
    False.
    Args:
        params (Dict): The parameter set for which trainable statuses should
            be derived from.
        trainables (Dict): A dictionary of boolean trainability statuses. The
            dictionary is equal in structure to the input params dictionary.
    Returns:
        Dict: A dictionary parameters. The dictionary is equal in structure to
            the input params dictionary.
    """

    def _stop_grad(p, t):
        return lax.cond(t, lambda x: x, lax.stop_gradient, p)


    return jtu.tree_map(lambda p, t: _stop_grad(p, t), obj, obj.trainables)


__all__ = [
    "Module",
    "param",
    "constrain",
    "unconstrain",
    "stop_gradients",
]
