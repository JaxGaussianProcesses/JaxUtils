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

from jaxutils.scan import vscan

import jax.numpy as jnp

# TODO: Thorough checks on vscan.
def test_vscan():
    def body(c, x):
        a, b = x
        return c, a + b

    xs = (jnp.arange(10), jnp.arange(10))
    c, ys = vscan(body, 0, xs)

    assert c == 0
    assert jnp.all(ys == jnp.arange(10) * 2)
