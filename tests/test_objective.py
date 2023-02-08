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
from jaxutils.objective import Objective
from jaxutils.module import Module
from jaxutils.dataset import Dataset

def test_objective() -> None:
    with pytest.raises(TypeError):
        Objective(negative=False)

    with pytest.raises(TypeError):
        Objective(negative=True)

    class DummyModel(Module):
        def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
            return x

    class DummyObjective(Objective):
        def evaluate(self, model: Module, train_data: Dataset) -> float:
            return jnp.sum((model(train_data.X) - train_data.y) ** 2)


    # Test intialisation
    objective = DummyObjective()
    model = DummyModel()
    data = Dataset(jnp.ones((10, 1)), jnp.ones((10, 1)))
    assert objective(model, data) == 0.0

    # Test negative
    data_new = Dataset(jnp.linspace(0, 1, 10).reshape(-1, 1), jnp.linspace(2, 5, 10).reshape(-1, 1))
    objective_negative = DummyObjective(negative=True)
    assert objective(model, data_new) + objective_negative(model, data_new) == 0.0