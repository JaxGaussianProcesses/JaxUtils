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

"""This non-public module defines PyTree node updating functionality for the `jaxutils.Module` metadata via the `jax.numpy` e.g. `.at` and `.set` syntax.

The following classes mimic what is done in lax.numpy.
"""

from __future__ import annotations
from typing import Callable, Any, Union, Sequence, Iterable
from jaxtyping import PyTree
import jax.tree_util as jtu


class _MetaUpdateRef:

    __slots__ = (
        "pytree",
        "where",
    )

    def __init__(self, pytree: PyTree, where: Callable[[PyTree], PyTree]) -> None:
        self.pytree = pytree
        self.where = where

    def __repr__(self) -> str:
        return f"_MetaUpdateRef({repr(self.pytree)}, {repr(self.where)})"

    def get(self, name: str = None, value: Any = None) -> PyTree:
        if name is None:
            return self.where(
                jtu.tree_structure(self.pytree).unflatten(self.pytree.__meta__)
            )

        else:
            return self.where(
                jtu.tree_structure(self.pytree).unflatten(
                    [meta_.get(name, value) for meta_ in self.pytree.__meta__]
                )
            )

    def set(self, **kwargs) -> PyTree:
        return _meta_at(where=self.where, pytree=self.pytree, replace=kwargs)

    def apply(self, func: Callable) -> PyTree:
        return _meta_at(where=self.where, pytree=self.pytree, replace_fn=func)


class _MetaUpdateHelper:
    """Helper class for updating a PyTree."""

    __slots__ = ("pytree",)

    def __init__(self, pytree: PyTree) -> None:
        self.pytree = pytree

    def __getitem__(self, where: Callable[[PyTree], PyTree]) -> _MetaUpdateRef:
        return _MetaUpdateRef(self.pytree, _to_callable(where))

    def __repr__(self) -> str:
        return f"_MetaUpdateHelper({repr(self.pytree)})"


def _to_callable(where: Union[Callable, Sequence[str], Ellipsis]) -> Callable:

    if isinstance(where, str):
        where = eval("lambda x: x." + where)

    if isinstance(where, Iterable):

        def _to_path(it: Iterable):
            return "".join(
                [str(elem) if not isinstance(elem, str) else "." + elem for elem in it]
            )

        where = eval("lambda x: x" + _to_path(where))

    if isinstance(where, type(Ellipsis)):
        where = lambda x: x

    return where


from typing import Any
import jax.tree_util as jtu

_Node = Any


class _LeafWrapper:
    __slots__ = ("value",)

    def __init__(self, value: Any):
        self.value = value


def _remove_leaf_wrapper(x: _LeafWrapper) -> Any:
    assert type(x) is _LeafWrapper
    return x.value


class _CountedIdDict:

    __slots__ = ("_dict",)

    def __init__(self, keys, values):
        assert len(keys) == len(values)
        self._dict = {id(k): (0, v) for k, v in zip(keys, values)}

    def __contains__(self, item):
        return id(item) in self._dict

    def __getitem__(self, item):
        c, v = self._dict[id(item)]
        self._dict[id(item)] = (c + 1, v)
        return v

    def get(self, item, default):
        try:
            return self[item]
        except KeyError:
            return default

    def count(self, item):
        return self._dict[id(item)][0]


def _meta_map(f: Callable[..., Any], tree: PyTree) -> PyTree:
    """Apply a function to a pytree where the first argument are the pytree leaves, and the second argument are the pytree metadata leaves."""
    leaves, treedef = jtu.tree_flatten(tree)
    meta_leaves = tree.__meta__
    all_leaves = [leaves] + [meta_leaves]
    return treedef.unflatten(f(*xs) for xs in zip(*all_leaves))


# TODO: Rerwite this function to exploit flattened meta pytree structure -> This would be faster than the current implementation that has to unflatten the meta pytree for each leaf.
def _meta_at(
    where,
    pytree,
    replace=None,
    replace_fn=None,
    is_leaf=lambda x: isinstance(x, dict),
):
    """Adapted from Equinox's `tree_at`."""

    # We need to specify a particular node in a PyTree.
    # This is surprisingly difficult to do! As far as I can see, pretty much the only
    # way of doing this is to specify e.g. `x.foo[0].bar` via `is`, and then pulling
    # a few tricks to try and ensure that the same object doesn't appear multiple
    # times in the same PyTree.
    #
    # So this first `tree_map` serves a dual purpose.
    # 1) Makes a copy of the composite nodes in the PyTree, to avoid aliasing via
    #    e.g. `pytree=[(1,)] * 5`. This has the tuple `(1,)` appear multiple times.
    # 2) It makes each leaf be a unique Python object, as it's wrapped in
    #    `_LeafWrapper`. This is needed because Python caches a few builtin objects:
    #    `assert 0 + 1 is 1`. I think only a few leaf types are subject to this.
    # So point 1) should ensure that all composite nodes are unique Python objects,
    # and point 2) should ensure that all leaves are unique Python objects.
    # Between them, all nodes of `pytree` are handled.
    #
    # I think pretty much the only way this can fail is when using a custom node with
    # singleton-like flatten+unflatten behaviour, which is pretty edge case. And we've
    # added a check for it at the bottom of this function, just to be sure.
    #
    # Whilst we're here: we also double-check that `where` is well-formed and doesn't
    # use leaf information. (As else `node_or_nodes` will be wrong.)
    pytree_old = pytree
    pytree = jtu.tree_structure(pytree).unflatten(pytree.__meta__)

    node_or_nodes_nowrapper = where(pytree)
    pytree = jtu.tree_map(_LeafWrapper, pytree, is_leaf=is_leaf)
    node_or_nodes = where(pytree)
    leaves1, structure1 = jtu.tree_flatten(node_or_nodes_nowrapper, is_leaf=is_leaf)
    leaves2, structure2 = jtu.tree_flatten(node_or_nodes)
    leaves2 = [_remove_leaf_wrapper(x) for x in leaves2]
    if (
        structure1 != structure2
        or len(leaves1) != len(leaves2)
        or any(l1 is not l2 for l1, l2 in zip(leaves1, leaves2))
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
        if replace is not None:
            replace = (replace,)
    else:
        nodes = node_or_nodes
    del in_pytree, node_or_nodes

    # Normalise replace vs replace_fn
    if replace is None:
        if replace_fn is None:
            raise ValueError(
                "Precisely one of `replace` and `replace_fn` must be specified."
            )
        else:

            def _replace_fn(x):
                x = jtu.tree_map(_remove_leaf_wrapper, x)
                return replace_fn(x)

            replace_fns = [_replace_fn] * len(nodes)
    else:
        if replace_fn is None:

            # TODO: Need to fix this.
            def _replace_fn(x, i):
                x = jtu.tree_map(_remove_leaf_wrapper, x)

                for k, v in replace.items():

                    # if len(nodes) != len(v):
                    #     raise ValueError(
                    #         "`where` must return a sequence of leaves of the same length as "
                    #         "`replace`."
                    #     )

                    try:
                        value = v[i]
                    except:
                        TypeError
                        value = v

                    x.update({k: value})

                return x

            replace_fns = [lambda x, i=i: _replace_fn(x, i) for i in range(len(nodes))]
        else:
            raise ValueError(
                "Precisely one of `replace` and `replace_fn` must be specified."
            )
    node_replace_fns = _CountedIdDict(nodes, replace_fns)

    # Actually do the replacement
    def _make_replacement(x: _Node) -> Any:
        return node_replace_fns.get(x, _remove_leaf_wrapper)(x)

    out = jtu.tree_map(
        _make_replacement, pytree, is_leaf=lambda x: x in node_replace_fns
    )

    print(out)

    new = pytree_old.__set_meta_func__(lambda _: jtu.tree_leaves(out, is_leaf=is_leaf))

    return new
