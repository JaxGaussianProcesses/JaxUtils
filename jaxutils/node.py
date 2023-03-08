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

"""This non-public module defines PyTree node updating functionality for the `jaxutils.Module` via the `jax.numpy` e.g. `.at` and `.set` syntax."""

from __future__ import annotations
from typing import Any, Callable, Sequence, TypeVar
import jax.tree_util as jtu
from .pytree import PyTree

NodeType = TypeVar("NodeType", bound=PyTree)


class _LeafWrapper:
    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    assert type(x) is _LeafWrapper
    return x.value


class _CountedIdDict:
    __slots__ = (
        "_dict",
        "_count",
    )

    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self._dict = {id(k): v for k, v in zip(keys, values)}
        self._count = {id(k): 0 for k in keys}

    def __contains__(self, item):
        return id(item) in self._dict

    def __getitem__(self, item):
        self._count[id(item)] += 1
        return self._dict[id(item)]

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def count(self, item):
        return self._count[id(item)]


# TODO: Clean up the inside code of this function.
def node_at(
    where: Callable[[NodeType], Sequence[NodeType]],
    pytree: PyTree,
    operator: Callable[[NodeType], Any] = None,
    node_type: NodeType = PyTree,
):
    """Update a PyTree node or nodes in-place.

    Args:
        where (Callable[[NodeType], Sequence[NodeType]]): A function that takes a node of type `node_type` and returns a sequence of nodes of type `node_type`.
        pytree (PyTree): The PyTree to update.
        replace (Sequence[Any]): A sequence of leaves to replace the nodes returned by `where`. Must be specified if `replace_fn` is not specified.
        replace_fn (Callable[[NodeType], Any]): A function that takes a node of type `node_type` and returns a leaf to replace the node. Must be specified if `replace` is not specified.
        node_type (NodeType): The type of the nodes to update. Defaults to `PyTree`.

    Returns:
        PyTree: The updated PyTree.

    Acknowledgements:
        Adapted from Equinox's `tree_at`.
    """
    is_leaf = lambda x: isinstance(x, node_type) and not pytree

    node_or_nodes_nowrapper = where(pytree)
    pytree = jtu.tree_map(_LeafWrapper, pytree, is_leaf=is_leaf)
    node_or_nodes = where(pytree)
    leaves1, structure1 = jtu.tree_flatten(node_or_nodes_nowrapper, is_leaf=is_leaf)
    leaves2, structure2 = jtu.tree_flatten(node_or_nodes)
    leaves2 = [_remove_leaf_wrapper(x) for x in leaves2]
    if (
        structure1 != structure2
        or len(leaves1) != len(leaves2)
        or any(
            l1 is not l2 and isinstance(l1, node_type)
            for l1, l2 in zip(leaves1, leaves2)
        )
    ):
        raise ValueError(
            "`where` must use just the PyTree structure of `pytree`. `where` must not "
            "depend on the leaves in `pytree`."
        )
    del node_or_nodes_nowrapper, leaves1, structure1, leaves2, structure2

    # Normalise whether we were passed a single node or a sequence of nodes.
    in_pytree = False

    def _in_pytree(x):
        nonlocal in_pytree
        if x is node_or_nodes:  # noqa: F821
            in_pytree = True
        return x  # needed for jax.tree_util.Partial, which has a dodgy constructor

    jtu.tree_map(_in_pytree, pytree, is_leaf=lambda x: x is node_or_nodes)  # noqa: F821

    if in_pytree:
        nodes = (node_or_nodes,)
    else:
        nodes = node_or_nodes

    del in_pytree, node_or_nodes

    def _replace_fn(x):
        x = jtu.tree_map(_remove_leaf_wrapper, x)
        return operator(x)

    replace_fns = [_replace_fn] * len(nodes)
    node_replace_fns = _CountedIdDict(nodes, replace_fns)

    # Actually do the replacement
    def _make_replacement(x: NodeType) -> Any:
        return node_replace_fns.get(x, _remove_leaf_wrapper)(x)

    out = jtu.tree_map(
        _make_replacement, pytree, is_leaf=lambda x: x in node_replace_fns
    )

    # Check that `where` is well-formed.
    for node in nodes:
        count = node_replace_fns.count(node)
        if count == 0:
            raise ValueError(
                "`where` does not specify an element or elements of `pytree`."
            )
        elif count == 1:
            pass
        else:
            raise ValueError(
                "`where` does not uniquely identify a single element of `pytree`. This "
                "usually occurs when trying to replace a `None` value:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True)\n"
                "\n"
                "\n"
                "for which the fix is to specify that `None`s should be treated as "
                "leaves:\n"
                "\n"
                "  >>> eqx.tree_at(lambda t: t[0], (None, None, 1), True,\n"
                "  ...             is_leaf=lambda x: x is None)"
            )

    return out
