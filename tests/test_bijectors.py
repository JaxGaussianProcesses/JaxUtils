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
from jaxutils.bijectors import Bijector, Identity, Softplus
from typing import Callable

@pytest.mark.parametrize("fwd", [lambda x: x, lambda x: jnp.log(x)])
@pytest.mark.parametrize("inv", [lambda x: x, lambda x: jnp.exp(x)])
def test_bijector(fwd: Callable, inv: Callable) -> None:
    b = Bijector(fwd, inv)
    assert b.forward(1.0) == fwd(1.0)
    assert b.inverse(1.0) == inv(1.0)

def test_identity() -> None:
    b = Identity
    assert b.forward(10.0) == 10.0
    assert b.inverse(29.0) == 29.0

def test_softplus() -> None:
    b = Softplus
    assert b.forward(10.0) == jnp.log(1.0 + jnp.exp(10.0))
    assert b.inverse(29.0) == jnp.log(jnp.exp(29.0) - 1.0)