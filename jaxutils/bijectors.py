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
"""Bijectors for use in model optimisation."""

import jax.numpy as jnp
import equinox as eqx
from typing import Callable


class Bijector(eqx.Module):
    """Base class for parameter bijector transformations. These are useful for model optimisation, where gradients are taken in the "unconstrained" (real) parameter space, while evaluating the model takes place in the "constrained" parameter space.

    For example, the Softplus bijector, f, is defined as:
        f(x) = log(1 + exp(x))
        f^{-1}(y) = log(exp(y) - 1)

    gives the "unconstrained" parameter space as the real line, while the "constrained" parameter space is the positive real line.
    This means we can ensure that the corresponding model parameter remains positive during optimisation.

    To implement your own bijector, you need to do is define a `forward` and `inverse` transformation!

    Adding log_det_jacobian's etc., is on the TODO list of this class.
    """

    forward: Callable = eqx.static_field()
    inverse: Callable = eqx.static_field()


"""Identity bijector."""
Identity = Bijector(forward=lambda x: x, inverse=lambda x: x)

"""Softplus bijector."""
Softplus = Bijector(
    forward=lambda x: jnp.log(1 + jnp.exp(x)),
    inverse=lambda x: jnp.log(jnp.exp(x) - 1.0),
)

"""Triangular bijector."""
# TODO: Add triangular bijector.


__all__ = [
    "Bijector",
    "Identity",
    "Softplus",
]
