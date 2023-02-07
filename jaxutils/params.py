# Copyright 2022 The JaxGaussianProcesses Contributors. All Rights Reserved.
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

import equinox as eqx
from dataclasses import fields, field
from dataclasses import field, fields
from typing import List

import jax.tree_util as jtu
from jax import lax

from .config import Identity
from .base import Base
from .bijector import Bijector


def build_bijectors(obj: Base) -> eqx.Module:
    """Given a Base object, return an equinox Module of bijectors comprising the same structure.

    Args:
        obj(Base): A Base object.
    
    Returns:
        eqx.Module: A Module of bijectors.
    """

    _, treedef = jtu.tree_flatten(obj)

    def _unpack_bijectors_from_meta(cls: eqx.Module):
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

                elif isinstance(value, eqx.Module):
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
        static: Whether the parameter is static or not.

    Returns:
        A field with the metadata set.
    """
    
    # # Create metadata dictionary.
    # try:
    #     metadata = dict(kwargs["metadata"])
    # except KeyError:
    #     metadata = kwargs["metadata"] = {}

    # if "param" in metadata:
    #     raise ValueError("Cannot use metadata with `param` already set.")
    # metadata["param"] = True


    # # Param has a transform.
    # if "transform" in metadata:
    #     raise ValueError("Cannot use metadata with `transform` already set.")
    # metadata["transform"] = transform

    return field(metadata={"transform": transform})


def constrain(obj: Base, bij: eqx.Module) -> Base:
    """
    Transform the parameters to the constrained space for corresponding
    bijectors.

    Args:
        obj (Base): The Base that is to be transformed.
        bij (Dict): The bijectors that are to be used for
            transformation.

    Returns:
        Base: A transformed parameter set. The dictionary is equal in
            structure to the input params dictionary.
    """
    #TODO: This will break if a leaf node is not a parameter, i.e., does not have a transform!
    return jtu.tree_map(lambda param, trans: trans.forward(param), obj, bij)


def unconstrain(obj: Base, bij: eqx.Module) -> Base:
    """
    Transform the parameters to the unconstrained space for corresponding
    bijectors.

    Args:
        obj (Base): The Base that is to be transformed.
        bij (Dict): The bijectors that are to be used for
            transformation.

    Returns:
        Base: A transformed model object.
    """
    #TODO: This will break if a leaf node is not a parameter, i.e., does not have a transform!
    return jtu.tree_map(lambda param, trans: trans.inverse(param), obj, bij)


def build_trainables(obj: Base, status: bool = True) -> eqx.Module:
    """
    Construct a dictionary of trainable statuses for each parameter. By default,
    every parameter within the model is trainable.

    Args:
        obj (Dict): The parameter set for which trainable statuses should be
            derived from.
        status (bool): The status of each parameter. Default is True.

    Returns:
        Dict: A dictionary of boolean trainability statuses. The dictionary is
            equal in structure to the input params dictionary.
    """
    return jtu.tree_map(lambda _: status, obj)


def trainable_params(module: eqx.Module, trainables: eqx.Module) -> eqx.Module:
    """
    Stop the gradients flowing through parameters whose trainable status is
    False.

    Args:
        module (eqx.Module): The equinox Module to set the trainability of.
        trainables (eqx.Module): The equinox Module of trainability statuses.

    Returns:
        eqx.Module: The equinox Module of parameters with stopped gradients.
    """

    return jtu.tree_map(lambda p, t: lax.cond(t, lambda x: x, lax.stop_gradient, p), module, trainables)


__all__ = [
    "param",
    "build_bijectors",
    "constrain",
    "unconstrain",
    "build_trainables",
    "trainable_params",
]
