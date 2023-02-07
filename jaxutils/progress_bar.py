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

from typing import Callable, Any, Union

from jax import lax
from jax.experimental import host_callback
from jaxtyping import Array, Float
from tqdm.auto import tqdm


#TODO: (Dan D) add a compilation message to the progress bar from your private code.

def progress_bar_scan(num_iters: int, log_rate: int) -> Callable:
    """Progress bar for Jax.lax scans (adapted from https://www.jeremiecoullon.com/2021/01/29/jax_progress_bar/)."""

    tqdm_bars = {}
    remainder = num_iters % log_rate

    def _define_tqdm(args: Any, transform: Any) -> None:
        """Define a tqdm progress bar."""
        tqdm_bars[0] = tqdm(range(num_iters))

    def _update_tqdm(args: Any, transform: Any) -> None:
        """Update the tqdm progress bar with the latest objective value."""
        loss_val, arg = args
        tqdm_bars[0].update(arg)
        tqdm_bars[0].set_postfix({"Objective": f"{loss_val: .2f}"})

    def _close_tqdm(args: Any, transform: Any) -> None:
        """Close the tqdm progress bar."""
        tqdm_bars[0].close()

    def _callback(cond: bool, func: Callable, arg: Any) -> None:
        """Callback a function for a given argument if a condition is true."""
        dummy_result = 0

        def _do_callback(_) -> int:
            """Perform the callback."""
            return host_callback.id_tap(func, arg, result=dummy_result)

        def _not_callback(_) -> int:
            """Do nothing."""
            return dummy_result

        _ = lax.cond(cond, _do_callback, _not_callback, operand=None)

    def _update_progress_bar(loss_val: Float[Array, "1"], iter_num: int) -> None:
        """Updates tqdm progress bar of a JAX scan or loop."""

        # Conditions for iteration number
        is_first: bool = iter_num == 0
        is_multiple: bool = (iter_num % log_rate == 0) & (
            iter_num != num_iters - remainder
        )
        is_remainder: bool = iter_num == num_iters - remainder
        is_last: bool = iter_num == num_iters - 1

        # Define progress bar, if first iteration
        _callback(is_first, _define_tqdm, None)

        # Update progress bar, if multiple of log_rate
        _callback(is_multiple, _update_tqdm, (loss_val, log_rate))

        # Update progress bar, if remainder
        _callback(is_remainder, _update_tqdm, (loss_val, remainder))

        # Close progress bar, if last iteration
        _callback(is_last, _close_tqdm, None)

    def _progress_bar_scan(body_fun: Callable) -> Callable:
        """Decorator that adds a progress bar to `body_fun` used in `lax.scan`."""

        def wrapper_progress_bar(carry: Any, x: Union[tuple, int]) -> Any:

            # Get iteration number
            if type(x) is tuple:
                iter_num, *_ = x
            else:
                iter_num = x

            # Compute iteration step
            result = body_fun(carry, x)

            # Get loss value
            *_, loss_val = result

            # Update progress bar
            _update_progress_bar(loss_val, iter_num)

            return result

        return wrapper_progress_bar

    return _progress_bar_scan


__all__ = [
    "progress_bar_scan",
]
