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
from jaxutils.module import Module, constrain, param, stop_gradients, unconstrain
from jaxutils.bijectors import Identity, Softplus
import jax.tree_util as jtu
import jax

from typing import Any

import jax.tree_util as jtu
import pytest
import equinox as eqx


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

    tree = Tree(
        param_a=1.0,
        sub_tree=SubTree(param_c=2.0, param_d=3.0, param_e=4.0),
        param_b=5.0,
    )

    assert isinstance(tree, Module)
    assert isinstance(tree.trainables[...].get(), Module)
    assert isinstance(tree.bijectors[...].get(), Module)
    assert isinstance(tree, eqx.Module)
    assert isinstance(tree.trainables[...].get(), eqx.Module)
    assert isinstance(tree.bijectors[...].get(), eqx.Module)

    assert tree.param_a == 1.0
    assert tree.sub_tree.param_c == 2.0
    assert tree.sub_tree.param_d == 3.0
    assert tree.sub_tree.param_e == 4.0
    assert tree.param_b == 5.0

    # Test default bijectors
    bijectors = tree.bijectors[...].get()
    bijector_list = jtu.tree_leaves(bijectors)

    for b1, b2 in zip(
        bijector_list, [Identity, Identity, Softplus, Softplus, Softplus]
    ):
        assert b1 == b2

    # Test default trainables
    trainables = tree.trainables[...].get()
    trainable_list = jtu.tree_leaves(trainables)

    for t1, t2 in zip(trainable_list, [True, True, True, True, True]):
        assert t1 == t2

    # Test constrain and unconstrain
    constrained = constrain(tree)
    unconstrained = unconstrain(tree)

    leafs = jtu.tree_leaves(tree)

    for l1, l2, bij in zip(
        leafs,
        jtu.tree_leaves(constrained),
        [Identity, Identity, Softplus, Softplus, Softplus],
    ):
        assert bij.forward(l1) == l2

    for l1, l2, bij in zip(
        leafs,
        jtu.tree_leaves(unconstrained),
        [Identity, Identity, Softplus, Softplus, Softplus],
    ):
        assert bij.inverse(l1) == l2

    _, tree_def = jtu.tree_flatten(tree)
    trainables = tree_def.unflatten([True, False, True, False, False])

    new_tree = tree.trainables[...].set(trainables)

    # Test stop gradients
    def loss(tree):
        tree = stop_gradients(tree)
        return jnp.sum(
            tree.param_a**2
            + tree.sub_tree.param_c**2
            + tree.sub_tree.param_d**2
            + tree.sub_tree.param_e**2
            + tree.param_b**2
        )

    g = jax.grad(loss)(new_tree)

    assert g.param_a == 2.0
    assert g.sub_tree.param_c == 0.0
    assert g.sub_tree.param_d == 6.0
    assert g.sub_tree.param_e == 0.0
    assert g.param_b == 0.0


# TODO: Fix these tests.
# def test_module_incorrect_typing():
#     class NotAModule:
#         pass

#     class SubTree(Module):
#         param_c: float = param(Identity)
#         param_d: float = param(Softplus)
#         param_e: float = param(Softplus)

#     class Tree(Module):
#         param_a: float = param(Identity)
#         sub_tree: NotAModule
#         param_b: float = param(Softplus)

#     with pytest.raises(ValueError):
#         Tree(
#             param_a=1.0,
#             sub_tree=SubTree(param_c=2.0, param_d=3.0, param_e=4.0),
#             param_b=5.0,
#         )


# def test_module_unmarked_param():
#     class SubTree(Module):
#         param_c: float
#         param_d: float = param(Softplus)
#         param_e: float = param(Softplus)

#     class Tree(Module):
#         param_a: float = param(Identity)
#         sub_tree: SubTree
#         param_b: float = param(Softplus)

#     with pytest.raises(ValueError):
#         Tree(
#             param_a=1.0,
#             sub_tree=SubTree(param_c=2.0, param_d=3.0, param_e=4.0),
#             param_b=5.0,
#       )


def test_tuple_attribute():
    class SubTree(Module):
        param_a: int = param(transform=Identity, default=1)
        param_b: int = param(transform=Softplus, default=2)
        param_c: int = param(transform=Identity, default=3, trainable=False)

    class Tree(Module):
        trees: tuple

    tree = Tree((SubTree(), SubTree(), SubTree()))

    assert len(tree.__meta__.trainables) == 9
    assert len(tree.__meta__.bijectors) == 9
    assert tree.__meta__.trainables == (
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    )
    assert tree.__meta__.bijectors == (
        Identity,
        Softplus,
        Identity,
        Identity,
        Softplus,
        Identity,
        Identity,
        Softplus,
        Identity,
    )


def test_list_attribute():
    class SubTree(Module):
        param_a: int = param(transform=Identity, default=1)
        param_b: int = param(transform=Softplus, default=2)
        param_c: int = param(transform=Identity, default=3, trainable=False)

    class Tree(Module):
        trees: list

    tree = Tree([SubTree(), SubTree(), SubTree()])

    assert len(tree.__meta__.trainables) == 9
    assert len(tree.__meta__.bijectors) == 9
    assert tree.__meta__.trainables == (
        True,
        True,
        False,
        True,
        True,
        False,
        True,
        True,
        False,
    )
    assert tree.__meta__.bijectors == (
        Identity,
        Softplus,
        Identity,
        Identity,
        Softplus,
        Identity,
        Identity,
        Softplus,
        Identity,
    )


def test_module_not_enough_attributes():
    class MyModule1(Module):
        weight: Any = param(Identity)

    with pytest.raises(TypeError):
        MyModule1()

    class MyModule2(Module):
        weight: Any = param(Identity)

        def __init__(self):
            pass

    with pytest.raises(ValueError):
        MyModule2()
    with pytest.raises(TypeError):
        MyModule2(1)


def test_module_too_many_attributes():
    class MyModule1(Module):
        weight: Any = param(Identity)

    with pytest.raises(TypeError):
        MyModule1(1, 2)

    class MyModule2(Module):
        weight: Any = param(Identity)

        def __init__(self, weight):
            self.weight = weight
            self.something_else = True

    with pytest.raises(AttributeError):
        MyModule2(1)


def test_module_setattr_after_init():
    class MyModule(Module):
        weight: Any = param(Identity)

    m = MyModule(1)
    with pytest.raises(AttributeError):
        m.asdf = True


def test_wrong_attribute():
    class MyModule(Module):
        weight: Any = param(Identity)

        def __init__(self, value):
            self.not_weight = value

    with pytest.raises(AttributeError):
        MyModule(1)


# The main part of this test is to check that __init__ works correctly.
def test_inheritance():
    # no custom init / no custom init

    class MyModule(Module):
        weight: Any = param(Identity)

    class MyModule2(MyModule):
        weight2: Any = param(Identity)

    m = MyModule2(1, 2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = MyModule2(1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    m = MyModule2(weight=1, weight2=2)
    assert m.weight == 1
    assert m.weight2 == 2
    with pytest.raises(TypeError):
        m = MyModule2(2, weight=2)

    # not custom init / custom init

    class MyModule3(MyModule):
        weight3: Any = param(Identity)

        def __init__(self, *, weight3, **kwargs):
            self.weight3 = weight3
            super().__init__(**kwargs)

    m = MyModule3(weight=1, weight3=3)
    assert m.weight == 1
    assert m.weight3 == 3

    # custom init / no custom init

    class MyModule4(Module):
        weight4: Any = param(Identity)

        def __init__(self, value4, **kwargs):
            self.weight4 = value4
            super().__init__(**kwargs)

    class MyModule5(MyModule4):
        weight5: Any = param(Identity)

    with pytest.raises(TypeError):
        m = MyModule5(value4=1, weight5=2)

    class MyModule6(MyModule4):
        pass

    m = MyModule6(value4=1)
    assert m.weight4 == 1

    # custom init / custom init

    class MyModule7(MyModule4):
        weight7: Any = param(Identity)

        def __init__(self, value7, **kwargs):
            self.weight7 = value7
            super().__init__(**kwargs)

    m = MyModule7(value4=1, value7=2)
    assert m.weight4 == 1
    assert m.weight7 == 2


def test_static_field():
    class MyModule(Module):
        field1: int = param(Identity)
        field2: int = eqx.static_field()
        field3: int = eqx.static_field(default=3)

    m = MyModule(1, 2)
    flat, treedef = jtu.tree_flatten(m)
    assert len(flat) == 1
    assert flat[0] == 1
    rm = jtu.tree_unflatten(treedef, flat)
    assert rm.field1 == 1
    assert rm.field2 == 2
    assert rm.field3 == 3


def test_wrap_method():
    class MyModule(Module):
        a: int = param(Identity)

        def f(self, b):
            return self.a + b

    m = MyModule(13)
    assert isinstance(m.f, jtu.Partial)
    flat, treedef = jtu.tree_flatten(m.f)
    assert len(flat) == 1
    assert flat[0] == 13
    assert jtu.tree_unflatten(treedef, flat)(2) == 15


def test_init_subclass():
    ran = []

    class MyModule(Module):
        def __init_subclass__(cls, **kwargs):
            super().__init_subclass__(**kwargs)
            ran.append(True)

    class AnotherModule(MyModule):
        pass

    assert ran == [True]
