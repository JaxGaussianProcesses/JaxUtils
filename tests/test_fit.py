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

from jaxutils.dataset import Dataset
from jaxutils.fit import fit
from jaxutils.parameters import Parameters
import jax.numpy as jnp
import jax.random as jr
import optax as ox
from simple_pytree import Pytree


def test_simple_linear_model():
    # (1) Create a dataset:
    X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
    y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape).reshape(-1, 1)
    D = Dataset(X, y)

    # (2) Define your model:
    class LinearModel(Pytree):
        def predict(self, params: Parameters, x):
            return params["weight"] * x + params["bias"]

        def init_params(self):
            return Parameters({"weight": 1.0, "bias": 1.0})

        def objective(self, params: Parameters, train_data: Dataset) -> float:
            return jnp.mean((train_data.y - self.predict(params, train_data.X)) ** 2)

    model = LinearModel()
    params = model.init_params()

    # (4) Train!
    trained_params = fit(
        params=params,
        objective=model.objective,
        train_data=D,
        optim=ox.sgd(0.001),
        num_iters=100,
    )

    assert len(trained_params.training_history) == 100
    assert isinstance(trained_params, Parameters)
    assert model.objective(trained_params, D) < model.objective(params, D)
