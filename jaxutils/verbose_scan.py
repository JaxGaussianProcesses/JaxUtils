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

from typing import Callable, List, Optional, Tuple, TypeVar

Carry = TypeVar("Carry")
X = TypeVar("X")
Y = TypeVar("Y")

from jaxutils import progress_bar
import jax
import jax.numpy as jnp

# TODO: Either remove Value printing altogether or make it an optional argument.


def verbose_scan(
    f: Callable[[Carry, X], Tuple[Carry, Y]],
    init: Carry,
    xs: X,
    length: Optional[int] = None,
    reverse: bool = False,
    unroll: int = 1,
    log_rate: int = 10,
) -> Tuple[Carry, List[Y]]:
    """Scan with verbose output.

    !!! example
        ```python
        import jax.numpy as jnp
        import jaxutils as ju

        def f(carry, x):
            return carry + x, carry + x

        init = 0
        xs = jnp.arange(10)
        carry, ys = ju.verbose_scan(f, init, xs)
        print(carry, ys)

        # 45 [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]

        # The above is equivalent to:
        carry, ys = jax.lax.scan(f, init, xs)
        print(carry, ys)

        # 45 [0, 1, 3, 6, 10, 15, 21, 28, 36, 45]
        ```

    Args:
        f (Callable[[Carry, X], Tuple[Carry, Y]]): A function that takes in a carry and an input and returns a tuple of a new carry and an output.
        init (Carry): The initial carry.
        xs (X): The inputs.
        length (Optional[int]): The length of the inputs. If None, then the length of the inputs is inferred.
        reverse (bool): Whether to scan in reverse.
        unroll (int): The number of iterations to unroll.
        log_rate (int): The rate at which to log the progress bar.

    Returns:
        Tuple[Carry, List[Y]]: A tuple of the final carry and the outputs.
    """

    length = len(xs)

    @progress_bar(num_iters=length, log_rate=log_rate)
    def body_fun(carry: Carry, iter_num_and_xs: Tuple[int, X]) -> Tuple[Carry, Y]:
        iter_num, x = iter_num_and_xs
        carry, y = f(carry, x)
        return carry, y

    carry, ys = jax.lax.scan(
        body_fun,
        init,
        (jnp.arange(length), xs),
        length=length,
        reverse=reverse,
        unroll=unroll,
    )
    return carry, ys


__all__ = [
    "verbose_scan",
]
