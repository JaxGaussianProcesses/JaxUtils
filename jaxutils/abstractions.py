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

from jax.random import KeyArray
from jaxtyping import Array, Float

from .params import constrain, stop_gradients, unconstrain
from .module import Module

from .data import Dataset
from .objective import Objective
from .progress_bar import progress_bar_scan


def fit(
    objective: Objective,
    model: Module,
    train_data: Dataset,
    optax_optim: ox.GradientTransformation,
    num_iters: Optional[int] = 100,
    log_rate: Optional[int] = 10,
    verbose: Optional[bool] = True,
) -> Tuple[Module, Array]:
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Args:
        objective (Objective): The objective function that we are optimising with respect to.
        model (eqx.Module): The model that is to be optimised.
        bijectors (eqx.Module): The bijectors that are to be used to transform the model parameters.
        trainables (eqx.Module): The trainables that are to be used to determine which parameters are to be optimised.
        optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.

    Returns:
        Tuple[Module, Array]: A Tuple comprising the optimised model and training history respectively.
    """

    # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
    def loss(model: Module) -> Float[Array, "1"]:
        model = stop_gradients(model)
        model = constrain(model)
        return objective(model, train_data)

    # Tranform model to unconstrained space
    model = unconstrain(model)

    # Initialise optimiser state
    opt_state = optax_optim.init(model)

    # Iteration loop numbers to scan over
    iter_nums = jnp.arange(num_iters)

    # Optimisation step
    def step(carry, iter_num: int):
        model, opt_state = carry
        loss_val, loss_gradient = jax.value_and_grad(loss)(model)
        updates, opt_state = optax_optim.update(loss_gradient, opt_state, model)
        model = ox.apply_updates(model, updates)
        carry = model, opt_state
        return carry, loss_val

    # Display progress bar if verbose is True
    if verbose:
        step = progress_bar_scan(num_iters, log_rate)(step)

    # Run the optimisation loop
    (model, _), history = jax.lax.scan(step, (model, opt_state), iter_nums)

    # Tranform final model to constrained space
    model = constrain(model)

    return model, history


# def fit_batches(
#     objective: Objective,
#     model: eqx.Module,
#     bijectors: eqx.Module,
#     trainables: eqx.Module,
#     train_data: Dataset,
#     optax_optim: ox.GradientTransformation,
#     key: KeyArray,
#     batch_size: int,
#     num_iters: Optional[int] = 100,
#     log_rate: Optional[int] = 10,
#     verbose: Optional[bool] = True,
# ) -> Tuple[eqx.Module, Array]:
#     """Abstracted method for fitting a GP model with mini-batches respect to a
#     supplied objective function.
#     Optimisers used here should originate from Optax.

#     Args:
#         objective (Objective): The objective function that we are optimising with respect to.
#         model (eqx.Module): The model that is to be optimised.
#         bijectors (eqx.Module): The bijectors that are to be used to transform the model parameters.
#         trainables (eqx.Module): The trainables that are to be used to determine which parameters are to be optimised.
#         train_data (Dataset): The training dataset.
#         optax_optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
#         key (KeyArray): The PRNG key for the mini-batch sampling.
#         batch_size (int): The batch_size.
#         num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
#         log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
#         verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.

#     Returns:
#         Tuple[eqx.Module, Array]: A Tuple comprising the optimised model and training history respectively.
#     """

#     # Define optimisation loss function on unconstrained space, with a stop gradient rule for trainables that are set to False
#     def loss(model: eqx.Module, batch: Dataset) -> Float[Array, "1"]:
#         model = trainable_params(model, trainables)
#         model = constrain(model, bijectors)
#         return objective(model, batch)

#     # Tranform model to unconstrained space
#     model = unconstrain(model, bijectors)

#     # Initialise optimiser state
#     opt_state = optax_optim.init(model)

#     # Mini-batch random keys and iteration loop numbers to scan over
#     keys = jr.split(key, num_iters)
#     iter_nums = jnp.arange(num_iters)

#     # Optimisation step
#     def step(carry, iter_num__and__key):
#         iter_num, key = iter_num__and__key
#         model, opt_state = carry

#         batch = get_batch(train_data, batch_size, key)

#         loss_val, loss_gradient = eqx.filter_value_and_grad(loss)(model, batch)
#         updates, opt_state = optax_optim.update(loss_gradient, opt_state, model)
#         model = eqx.apply_updates(model, updates)

#         carry = model, opt_state
#         return carry, loss_val

#     # Display progress bar if verbose is True
#     if verbose:
#         step = progress_bar_scan(num_iters, log_rate)(step)

#     # Run the optimisation loop
#     (model, _), history = jax.lax.scan(step, (model, opt_state), (iter_nums, keys))

#     # Tranform final params to constrained space
#     model = constrain(model, bijectors)

#     return model, history


def get_batch(train_data: Dataset, batch_size: int, key: KeyArray) -> Dataset:
    """Batch the data into mini-batches. Sampling is done with replacement.

    Args:
        train_data (Dataset): The training dataset.
        batch_size (int): The batch size.

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample data inidicies with replacement to get the mini-batch
    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])


__all__ = [
    "fit",
    "fit_natgrads",
    "get_batch",
]