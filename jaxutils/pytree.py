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

"""This is based on the simple-pytree package by @cgarciae. Adapted to use static field convention of Equinox, and added metadata functionality."""

from __future__ import annotations

import dataclasses
from dataclasses import field, Field
from abc import ABCMeta
from typing import Any, Set, Dict, List
from copy import copy, deepcopy

import jax
import jax.tree_util as jtu


def static(**kwargs: Any) -> Field:
    """Used for marking that a field should _not_ be treated as a leaf of the PyTree.

    Equivalent to `equinox.static_field`.

    Args:
        **kwargs (Any): If any are passed then they are passed on to `dataclass.field`.
    """
    try:
        metadata = dict(kwargs["metadata"])
    except KeyError:
        metadata = kwargs["metadata"] = {}
    if "static" in metadata:
        raise ValueError("Cannot use metadata with `static` already set.")
    metadata["static"] = True
    return field(**kwargs)


class PyTreeMeta(ABCMeta):
    """Metaclass for PyTree."""

    def __call__(self, *args: Any, **kwds: Any):
        obj: PyTree = super().__call__(*args, **kwds)
        object.__setattr__(obj, "_pytree__initialized", True)
        return obj


class PyTree(metaclass=PyTreeMeta):
    """Base class for PyTree."""

    _pytree__initialized: bool
    _pytree__static_fields: Set[str]
    _pytree__class_is_mutable: bool
    _pytree__meta: Dict[str, Any]
    _pytree__annotations: List[str]
    _pytree__is_leaf: Dict[str:bool]

    def __init_subclass__(cls, mutable: bool = False):

        jtu.register_pytree_node(
            cls,
            flatten_func=tree_flatten,
            unflatten_func=lambda *_args: tree_unflatten(cls, *_args),
        )

        class_annotations = _get_all_annotations(cls)

        # init class variables
        cls._pytree__initialized = False  # initialize mutable
        cls._pytree__static_fields = set()
        cls._pytree__class_is_mutable = mutable
        cls._pytree__meta = {}
        cls._pytree__annotations = class_annotations
        cls._pytree__is_leaf = {}

        # get class info
        class_vars = _get_all_class_vars(cls)

        for field, value in class_vars.items():

            if "_pytree__" in field or (
                isinstance(value, dataclasses.Field)
                and value.metadata is not None
                and value.metadata.get("static", False)
            ):

                cls._pytree__static_fields.add(field)

            elif isinstance(value, dataclasses.Field) and value.metadata is not None:
                cls._pytree__meta[field] = {**value.metadata}

        for field in class_annotations:
            if "_pytree__" not in field and (
                field not in cls._pytree__static_fields
                and field not in cls._pytree__meta.keys()
            ):
                cls._pytree__meta[field] = {}

    def _replace_meta(self, **kwargs: Any) -> PyTree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        for key in kwargs:
            if key not in self._pytree__meta.keys():
                raise ValueError(f"'{key}' is not a leaf of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(_pytree__meta={**pytree._pytree__meta, **kwargs})
        return pytree

    def _update_meta(self, **kwargs: Any) -> PyTree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        for key in kwargs:
            if key not in self._pytree__meta.keys():
                raise ValueError(
                    f"'{key}' is not an attribute of {type(self).__name__}"
                )

        # TODO: Simplify.
        pytree = copy(self)
        new = deepcopy(pytree._pytree__meta)
        for key in kwargs:
            if key in new:
                new[key].update(kwargs[key])
            else:
                new[key] = kwargs[key]
        pytree.__dict__.update(_pytree__meta=new)
        return pytree

    def _replace(self: PyTree, **kwargs: PyTree) -> PyTree:
        """
        Replace the values of the fields of the object with the values of the
        keyword arguments. If the object is a dataclass, `dataclasses.replace`
        will be used. Otherwise, a new object will be created with the same
        type as the original object.
        """
        if dataclasses.is_dataclass(self):
            return dataclasses.replace(self, **kwargs)

        fields = vars(self)
        for key in kwargs:
            if key not in fields:
                raise ValueError(f"'{key}' is not a field of {type(self).__name__}")

        pytree = copy(self)
        pytree.__dict__.update(kwargs)
        return pytree

    def __setattr__(self, field: str, value: Any):

        if field not in self._pytree__annotations:
            raise AttributeError(f"{type(self)} has no annotation field {field}.")

        if self._pytree__initialized and not self._pytree__class_is_mutable:
            raise AttributeError(
                f"{type(self)} is immutable, trying to update field {field}."
            )

        _field_is_static = field in self._pytree__static_fields
        _value_is_pytree = (
            jtu.tree_map(
                lambda x: isinstance(x, PyTree),
                value,
                is_leaf=lambda x: isinstance(x, PyTree),
            )
            == False
        )
        _is_leaf = _value_is_pytree and not _field_is_static
        object.__setattr__(
            self, "_pytree__is_leaf", {**self._pytree__is_leaf, field: _is_leaf}
        )
        object.__setattr__(self, field, value)

    @property
    def pytree_at(self):
        from ._pytree import _PyTreeNodeUpdateHelper

        return _PyTreeNodeUpdateHelper(self)


def tree_flatten(pytree: PyTree):
    node_fields = {}
    static_fields = {}

    for field, value in vars(pytree).items():
        if field in pytree._pytree__static_fields:
            static_fields[field] = value
        else:
            node_fields[field] = value

    return (node_fields,), static_fields


def tree_unflatten(cls: PyTree, static_fields, children):
    """
    Unflattens a PyTree.

    Args:
        cls (PyTree): Class to unflatten.

    Returns:
        PyTree: Unflattened PyTree.
    """
    (node_fields,) = children
    pytree = cls.__new__(cls)
    pytree.__dict__.update(node_fields, **static_fields)
    return pytree


def _get_all_class_vars(cls: type) -> Dict[str, Any]:
    """
    Returns a dictionary of all the class variables of a class.

    Args:
        cls (type): Class to get the class variables of.

    Returns:
        Dict[str, Any]: Dictionary of all the class variables of the class.
    """
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__dict__"):
            d.update(vars(c))
    return d


def _get_all_annotations(cls: type) -> Dict[str, type]:
    """
    Returns a dictionary of all the annotations of a class.

    Args:
        cls (type): Class to get the annotations of.

    Returns:
        Dict[str, type]: Dictionary of all the annotations of the class.
    """
    d = {}
    for c in reversed(cls.mro()):
        if hasattr(c, "__annotations__"):
            d.update(**c.__annotations__)
    return d


def _unpack_metatree_leaves(pytree: PyTree) -> List[Dict[str, Any]]:
    """
    Returns a list of the PyTree leaves' metadata.

    Args:
        pytree (PyTree): PyTree to get the metadata of the leaves.

    Returns:
        List[Dict[str, Any]]: List of the PyTree leaves' metadata.
    """
    _leaf_metadata = [
        pytree._pytree__meta[k]
        for k, _is_leaf_bool in pytree._pytree__is_leaf.items()
        if _is_leaf_bool
    ]

    def _nested_unpack_metadata(pytree: PyTree, *rest: PyTree) -> None:
        if isinstance(pytree, PyTree):
            _leaf_metadata.extend(
                [
                    pytree._pytree__meta[k]
                    for k, _is_leaf_bool in pytree._pytree__is_leaf.items()
                    if _is_leaf_bool
                ]
            )
            _unpack_metadata(pytree, *rest)

    def _unpack_metadata(pytree: PyTree, *rest: PyTree) -> None:
        pytrees = (pytree,) + rest
        _ = jax.tree_map(
            _nested_unpack_metadata,
            *pytrees,
            is_leaf=lambda x: isinstance(x, PyTree) and not x in pytrees,
        )

    _unpack_metadata(pytree)

    return _leaf_metadata
