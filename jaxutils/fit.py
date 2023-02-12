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

from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.random as jr
import optax as ox

import jax.tree_util as jtu
from jax.random import KeyArray
from jax._src.random import _check_prng_key
from jaxtyping import Array, Float
from typing import Any

from .module import Module, constrain, unconstrain, stop_gradients
from .dataset import Dataset
from .bijectors import Bijector
from .objective import Objective
from .progress_bar import progress_bar_scan


def fit(
    *,
    model: Module,
    objective: Objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    key: Optional[KeyArray] = jr.PRNGKey(42),
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> Tuple[Module, Array]:
    """Train a Module model with respect to a supplied Objective function. Optimisers used here should originate from Optax.

    !!! example
        ```python
        import jax.numpy as jnp
        import jaxutils as ju
        import optax as ox

        # (1) Create a dataset:
        X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
        y = 2.0 * X + 1.0 + 10 * jr.normal(jr.PRNGKey(0), X.shape).reshape(-1, 1)
        D = ju.Dataset(X, y)

        # (2) Define your model:
        class LinearModel(ju.Module):
            weight: float = ju.param(transform=ju.Identity, trainable=True)
            bias: float = ju.param(transform=ju.Identity, trainable=True)

            def __call__(self, x):
                return self.weight * x + self.bias

        model = LinearModel(weight=1.0, bias=1.0)

        # (3) Define your loss function:
        class MeanSqaureError(Objective):
            def evaluate(self, model: LinearModel, train_data: ju.Dataset) -> float:
                y_pred = model(train_data.X)
                return jnp.mean((y_pred - train_data.y) ** 2)

        loss = MeanSqaureError()

        # (4) Define your optimiser:
        optim = ox.adam(1e-3)

        # (5) Train your model:
        model, history = ju.fit(model=model, objective=loss, train_data=D, optim=optim, num_iters=1000)

        # (6) Plot the training history:
        import matplotlib.pyplot as plt
        plt.plot(history)
        plt.show()

        # (7) Plot the model predictions:
        X_test = jnp.linspace(0.0, 10.0, 1000).reshape(-1, 1)
        y_test = model(X_test)
        plt.plot(X_test, y_test)
        plt.scatter(D.X, D.y)
        plt.show()

        # (8) Print the final model parameters:
        print(model)
        ```

    Args:
        model (Module): The model Module to be optimised.
        objective (Objective): The objective function that we are optimising with respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1 (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.

    Returns:
        Tuple[Module, Array]: A Tuple comprising the optimised model and training history respectively.
    """

    # Check inputs.
    _check_model(model)
    _check_objective(objective)
    _check_train_data(train_data)
    _check_optim(optim)
    _check_num_iters(num_iters)
    _check_batch_size(batch_size)
    _check_prng_key(key)
    _check_log_rate(log_rate)
    _check_verbose(verbose)

    # Unconstrained space loss function with stop-gradient rule for non-trainable params.
    def loss(model: Module, batch: Dataset) -> Float[Array, "1"]:
        model = stop_gradients(model)
        model = constrain(model)
        return objective(model, batch)

    # Unconstrained space model.
    model = unconstrain(model)

    # Initialise optimiser state.
    state = optim.init(model)

    # Mini-batch random keys and iteration loop numbers to scan over.
    iter_keys = jr.split(key, num_iters)
    iter_nums = jnp.arange(num_iters)

    # Optimisation step.
    def step(carry, iter_num__and__key):
        _, key = iter_num__and__key
        model, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(model, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, model)
        model = ox.apply_updates(model, updates)

        carry = model, opt_state
        return carry, loss_val

    # Progress bar, if verbose True.
    if verbose:
        step = progress_bar_scan(num_iters, log_rate)(step)

    # Optimisation loop.
    (model, _), history = jax.lax.scan(step, (model, state), (iter_nums, iter_keys))

    # Constrained space.
    model = constrain(model)

    return model, history


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.
        key (KeyArray): The random key to use for the batch selection.

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indicies with replacement.
    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])


def _check_model(model: Any) -> None:
    """Check that the model is of type Module. Check trainables and bijectors tree structure."""
    if not isinstance(model, Module):
        raise TypeError("model must be of type jaxutils.Module")

    if not jtu.tree_structure(model) == jtu.tree_structure(model.trainables):
        raise TypeError("trainables should have same tree structure as model")

    def _is_bij(x):
        return isinstance(x, Bijector)

    if not jtu.tree_structure(
        jtu.tree_map(lambda _: True, model.bijectors, is_leaf=_is_bij)
    ) == jtu.tree_structure(model):
        raise ValueError("bijectors tree must have the same structure as the Module.")


def _check_objective(objective: Any) -> None:
    """Check that the objective is of type Objective."""
    if not isinstance(objective, Objective):
        raise TypeError("objective must be of type jaxutils.Objective")


def _check_train_data(train_data: Any) -> None:
    """Check that the train_data is of type Dataset."""
    if not isinstance(train_data, Dataset):
        raise TypeError("train_data must be of type jaxutils.Dataset")


def _check_optim(optim: Any) -> None:
    """Check that the optimiser is of type GradientTransformation."""
    if not isinstance(optim, ox.GradientTransformation):
        raise TypeError("optax_optim must be of type optax.GradientTransformation")


def _check_num_iters(num_iters: Any) -> None:
    """Check that the number of iterations is of type int and positive."""
    if not isinstance(num_iters, int):
        raise TypeError("num_iters must be of type int")

    if not num_iters > 0:
        raise ValueError("num_iters must be positive")


def _check_log_rate(log_rate: Any) -> None:
    """Check that the log rate is of type int and positive."""
    if not isinstance(log_rate, int):
        raise TypeError("log_rate must be of type int")

    if not log_rate > 0:
        raise ValueError("log_rate must be positive")


def _check_verbose(verbose: Any) -> None:
    """Check that the verbose is of type bool."""
    if not isinstance(verbose, bool):
        raise TypeError("verbose must be of type bool")


def _check_batch_size(batch_size: Any) -> None:
    """Check that the batch size is of type int and positive if not minus 1."""
    if not isinstance(batch_size, int):
        raise TypeError("batch_size must be of type int")

    if not batch_size == -1:
        if not batch_size > 0:
            raise ValueError("batch_size must be positive")


__all__ = [
    "fit",
    "get_batch",
]
