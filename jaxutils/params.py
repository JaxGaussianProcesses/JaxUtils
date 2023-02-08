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

from dataclasses import field

import jax.tree_util as jtu
from jax import lax

from .bijectors import Bijector
from .module import Module


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
    Stop gradients flowing through parameters whose correponding leaf node status in the trainables PyTree is
    False.

    Args:
        module (Module): The jaxutils Module to set the trainability of.

    Returns:
        Module: The jaxutils Module of parameters with stopped gradients.
    """

    return jtu.tree_map(lambda p, t: lax.cond(t, lambda x: x, lax.stop_gradient, p), obj, obj.trainables)


__all__ = [
    "param",
    "constrain",
    "unconstrain",
    "stop_gradients",
]
