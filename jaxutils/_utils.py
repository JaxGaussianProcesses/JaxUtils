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

from typing import Sequence, Callable, Union, Iterable, Any


def _to_callable(where: Union[Callable, Sequence[str], Ellipsis]) -> Callable:
    """Convert a where argument to a callable function.

    Args:
        where (Union[Callable[[Union[Module, PyTree]], Sequence[Union[Module, PyTree]]], Sequence[str], Ellipsis]): Where argument.

    Returns:

        Callable[[Union[Module, PyTree]], Sequence[Union[Module, PyTree]]]]: Callable function.
    """

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
