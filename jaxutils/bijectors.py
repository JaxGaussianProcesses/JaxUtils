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

import distrax as dx
import jax.numpy as jnp
import tensorflow_probability.substrates.jax as tfb

Identity = dx.Lambda(forward=lambda x: x, inverse=lambda x: x)

Softplus = dx.Lambda(
    forward=lambda x: jnp.log(1 + jnp.exp(x)),
    inverse=lambda x: jnp.log(jnp.exp(x) - 1.0),
)

FillScaleTriL = dx.Chain(
    [
        tfb.FillScaleTriL(diag_shift=jnp.array(1e-6)),
    ]
)
