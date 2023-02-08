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
from typing import Any

from .params import constrain, stop_gradients, unconstrain
from .module import Module

from .data import Dataset
from .objective import Objective
from .progress_bar import progress_bar_scan



def fit(
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
    """Abstracted method for fitting a GP model with respect to a supplied objective function.
    Optimisers used here should originate from Optax.

    Args:
        objective (Objective): The objective function that we are optimising with respect to.
        model (eqx.Module): The model that is to be optimised.
        bijectors (eqx.Module): The bijectors that are to be used to transform the model parameters.
        trainables (eqx.Module): The trainables that are to be used to determine which parameters are to be optimised.
        optim (GradientTransformation): The Optax optimiser that is to be used for learning a parameter set.
        num_iters (Optional[int]): The number of optimisation steps to run. Defaults to 100.
        batch_size (Optional[int]): The size of the mini-batch to use. Defaults to -1 (i.e. full batch).
        key (Optional[KeyArray]): The random key to use for the optimisation batch selection. Defaults to jr.PRNGKey(42).
        log_rate (Optional[int]): How frequently the objective function's value should be printed. Defaults to 10.
        verbose (Optional[bool]): Whether to print the training loading bar. Defaults to True.

    Returns:
        Tuple[Module, Array]: A Tuple comprising the optimised model and training history respectively.
    """

    _check_types(model, objective, train_data, optim, num_iters, log_rate, verbose, None, None)

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


    # Optimisation step
    def step(carry, iter_num__and__key):
        iter_num, key = iter_num__and__key
        model, opt_state = carry

        if batch_size == -1:
            batch = train_data

        if batch_size != -1:
            batch = get_batch(train_data, batch_size, key)

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

    Returns:
        Dataset: The batched dataset.
    """
    x, y, n = train_data.X, train_data.y, train_data.n

    # Subsample mini-batch indicies with replacement.
    indicies = jr.choice(key, n, (batch_size,), replace=True)

    return Dataset(X=x[indicies], y=y[indicies])



def _check_types(
    model: Any, 
    objective: Any, 
    train_data: Any, 
    optax_optim: Any,
    num_iters: Any,
    log_rate: Any,
    verbose: Any,
    key: Any,
    batch_size: Any,
    ) -> None:

    if not isinstance(model, Module):
        raise TypeError("model must be of type jaxutils.Module")
    
    if not isinstance(objective, Objective):
        raise TypeError("objective must be of type jaxutils.Objective")
    
    if not isinstance(train_data, Dataset):
        raise TypeError("train_data must be of type jaxutils.Dataset")

    if not isinstance(optax_optim, ox.GradientTransformation):
        raise TypeError("optax_optim must be of type optax.GradientTransformation")

    if not isinstance(num_iters, int):
        raise TypeError("num_iters must be of type int")

    if not num_iters >  0:
        raise ValueError("num_iters must be positive, but got num_iters={num_iters}")

    if not isinstance(log_rate, int):
        raise TypeError(f"log_rate must be of type int")

    if not log_rate >  0:
        raise ValueError(f"log_rate must be positive, but got log_rate={log_rate}")

    if not isinstance(verbose, bool):
        raise TypeError("verbose must be of type bool")

    if key is not None:
        if not isinstance(key, KeyArray):
            raise TypeError("key must be of type jax.random.KeyArray")
    
    if batch_size is not None:
        if not isinstance(batch_size, int):
            raise TypeError("batch_size must be of type int")

        if not batch_size >  0:
            raise ValueError(f"batch_size must be positive, but got batch_size={batch_size}")
        
        if not batch_size < train_data.n:
            raise ValueError(f"batch_size must be less than train_data.n, but got batch_size={batch_size} and train_data.n={train_data.n}")


__all__ = [
    "fit",
    "fit_natgrads",
    "get_batch",
]