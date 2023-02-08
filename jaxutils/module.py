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

import jax.tree_util as jtu
from dataclasses import fields
from typing import List


import equinox as eqx
from .bijectors import Bijector


class _cache_property_comprising_static_fields:
    """Be careful with this --- highly JAX incompatible!!

    Courtesy of Kernex.

    Fine for properties that depend *only* on static fields. For dynamic fields, this will end in tears.
    """

    def __init__(self, prop):
        """ Does your prop have a dynamic field? If so, STOP!"""

        self.name = prop.__name__
        self.prop = prop

    def __get__(self, instance, owner):
        value = self.prop(instance)
        object.__setattr__(instance, self.name, value)
        return value


class Module(eqx.Module):
    """Base class for all objects of the JaxGaussianProcesses ecosystem.""" 

    # TODO: Add checks for param fields. WE NEED TO ENSURE ALL LEAVES ARE PARAMS AND ONLY PARAMS.

    def set_bijectors(self, bijectors):

        # TODO: Throwin hostcall back warning. This completely breaks the immutability of the object.
        # YOU DO NOT WANT TO DO THIS WITHIN A JIT REGION.
        # Really, this is a hack. We need to create a new object not modify the existing one.
        object.__setattr__(self, "bijectors", bijectors)

    def set_trainables(self, trainables):

        # TODO: Throwin hostcall back warning. This completely breaks the immutability of the object.
        # YOU DO NOT WANT TO DO THIS WITHIN A JIT REGION.
        # Really, this is a hack. We need to create a new object not modify the existing one.
        object.__setattr__(self, "trainables", trainables)


    @_cache_property_comprising_static_fields
    def bijectors(self):
        """Return a default PyTree of model bijectors."""
        return _default_bijectors(self)

    @_cache_property_comprising_static_fields
    def trainables(self):
        return _default_trainables(self)




def _default_trainables(obj: Module) -> eqx.Module:
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
    return jtu.tree_map(lambda _: True, obj)


def _default_bijectors(obj: Module) -> eqx.Module:
    """Given a Base object, return an equinox Module of bijectors comprising the same structure.

    Args:
        obj(Base): A Base object.
    
    Returns:
        eqx.Module: A Module of bijectors.
    """

    _, treedef = jtu.tree_flatten(obj)

    def _unpack_bijectors_from_meta(cls: eqx.Module) -> List[Bijector]:
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




__all__ = [
    "Module",
]
