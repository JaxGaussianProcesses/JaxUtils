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

import jax.numpy as jnp
import pytest
from jaxutils.module import Module, param, constrain, unconstrain, stop_gradients
from jaxutils.bijectors import Bijector, Identity, Softplus
import jax.tree_util as jtu
import jax

def test_module():

    # Test init
    class SubTree(Module):
        param_c: float = param(Identity)
        param_d: float = param(Softplus)
        param_e: float = param(Softplus)

    class Tree(Module):
        param_a: float = param(Identity)
        sub_tree: SubTree
        param_b: float = param(Softplus)

    tree = Tree(param_a=1.0, sub_tree=SubTree(param_c=2.0, param_d=3.0, param_e=4.0), param_b=5.0)

    assert tree.param_a == 1.0
    assert tree.sub_tree.param_c == 2.0
    assert tree.sub_tree.param_d == 3.0
    assert tree.sub_tree.param_e == 4.0
    assert tree.param_b == 5.0


    bijector_list = jtu.tree_leaves(tree.bijectors)

    for b1, b2 in zip(bijector_list, [Identity, Identity, Softplus, Softplus, Softplus]):
        assert b1 == b2

    trainable_list = jtu.tree_leaves(tree.trainables)

    for t1, t2 in zip(trainable_list, [True, True, True, True, True]):
        assert t1 == t2

    # Test constrain and unconstrain
    constrained = constrain(tree)
    unconstrained = unconstrain(tree)
    
    leafs = jtu.tree_leaves(tree)

    for l1, l2, bij in zip(leafs, jtu.tree_leaves(constrained), [Identity, Identity, Softplus, Softplus, Softplus]):
        assert bij.forward(l1) == l2

    for l1, l2, bij in zip(leafs, jtu.tree_leaves(unconstrained), [Identity, Identity, Softplus, Softplus, Softplus]):
        assert bij.inverse(l1) == l2


    _, tree_def = jax.tree_flatten(tree)
    tree = tree.set_trainables(tree_def.unflatten([True, False, True, False, False]))


    # Test stop gradients
    def loss(tree):
        tree = stop_gradients(tree)
        return jnp.sum(tree.param_a**2 + tree.sub_tree.param_c**2 + tree.sub_tree.param_d**2 + tree.sub_tree.param_e**2 + tree.param_b**2)

    g = jax.grad(loss)(tree)

    assert g.param_a == 2.0
    assert g.sub_tree.param_c == 4.0
    assert g.sub_tree.param_d == 6.0
    assert g.sub_tree.param_e == 8.0
    assert g.param_b == 10.0


