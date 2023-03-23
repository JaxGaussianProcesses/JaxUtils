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

from typing import Callable, Optional

import jax
import jax.random as jr
import optax as ox

from jax.random import KeyArray
from jax._src.random import _check_prng_key
from jaxtyping import Array, Float
from typing import Any

from .parameters import Parameters
from .dataset import Dataset
from .scan import vscan


def fit(
    *,
    objective,
    train_data: Dataset,
    optim: ox.GradientTransformation,
    params: Parameters = None,
    fn: Callable[[Parameters, Dataset], Float[Array, "1"]] = None,
    num_iters: Optional[int] = 100,
    batch_size: Optional[int] = -1,
    key: Optional[KeyArray] = jr.PRNGKey(42),
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
    unroll: int = 1,
) -> Parameters:
    """Train a Module model with respect to a supplied Objective function. Optimisers
    used here should originate from Optax.

    Args:
        params (Parameters): The parameters to be optimised.
        objective (Callable[[Parameters, Dataset], Float[Array, "1"]]): The objective
            function that we are optimising with respect to.
        train_data (Dataset): The training data to be used for the optimisation.
        optim (GradientTransformation): The Optax optimiser that is to be used for
            learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to
            100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1
            (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch
            selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should
            be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults
            to True.
        unroll (int): The number of unrolled steps to use for the optimisation.
            Defaults to 1.

    Returns:
        Parameters: A Tuple comprising the optimised model and training
            history respectively.
    """
    if params is None:
        params = objective.init_params(key)

    if fn is None:
        fn = jax.jit(objective.step)

    # Check inputs.
    _check_train_data(train_data)
    _check_optim(optim)
    _check_num_iters(num_iters)
    _check_batch_size(batch_size)
    _check_prng_key(key)
    _check_log_rate(log_rate)
    _check_verbose(verbose)

    # Unconstrained space loss fn. with stop-gradient rule for non-trainable params.
    def loss(params: Parameters, batch: Dataset) -> Float[Array, "1"]:
        params = params.stop_gradients()
        return fn(params.constrain(), batch)

    # Unconstrained space params.
    params = params.unconstrain()

    # Initialise optimiser state.
    state = optim.init(params)

    # Mini-batch random keys to scan over.
    iter_keys = jr.split(key, num_iters)

    # Optimisation step.
    def step(carry, key):
        params, opt_state = carry

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)
        else:
            batch = train_data

        loss_val, loss_gradient = jax.value_and_grad(loss)(params, batch)
        updates, opt_state = optim.update(loss_gradient, opt_state, params)
        params = ox.apply_updates(params, updates)

        carry = params, opt_state
        return carry, loss_val

    # Optimisation scan.
    scan = vscan if verbose else jax.lax.scan

    # Optimisation loop.
    (params, _), history = scan(step, (params, state), (iter_keys), unroll=unroll)

    # Constrained space.
    params = params.constrain()
    params = params.update_training_history(history)

    return params


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
