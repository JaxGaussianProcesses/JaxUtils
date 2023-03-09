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

from __future__ import annotations

import jax.tree_util as jtu
import jax
from typing import Dict, Any
from .bijectors import Identity
from simple_pytree import Pytree, static_field


class Parameters(Pytree, dict):
    """
    The state of the model. This includes the parameter set, which parameters
    are to be trained and bijectors that allow parameters to be constrained and
    unconstrained.
    """

    _param_dict: Dict
    _bijector_dict: Dict = static_field()
    _trainable_dict: Dict = static_field()
    _training_history: list = static_field()

    def __init__(
        self,
        params: Dict,
        bijectors: Dict = None,
        trainables: Dict = None,
        training_history: list = None,
    ):

        if bijectors is None:
            bijectors = jtu.tree_map(lambda _: Identity, params)

        if trainables is None:
            trainables = jtu.tree_map(lambda _: True, params)

        self._param_dict = params
        self._trainable_dict = trainables
        self._bijector_dict = bijectors
        self._training_history = training_history

    def __repr__(self) -> str:
        return f"Parameters({self.params.__repr__()})"

    def __getitem__(self, __name: str) -> Any:
        return self._param_dict.__getitem__(__name)

    def __setitem__(self, __name: str, __value: Any) -> None:
        return self._param_dict.__setitem__(__name, __value)

    @property
    def params(self) -> Dict:
        return self._param_dict

    def update_params(self, value: Dict) -> Parameters:
        return Parameters(value, self.bijectors, self.trainables, self.training_history)

    @property
    def bijectors(self) -> Dict:
        return self._bijector_dict

    def update_bijectors(self, value: Dict) -> Parameters:
        return Parameters(self.params, value, self.trainables, self.training_history)

    @property
    def trainables(self) -> Dict:
        return self._trainable_dict

    def update_trainables(self, value: Dict) -> Parameters:
        return Parameters(self.params, self.bijectors, value, self.training_history)

    @property
    def training_history(self) -> list:
        return self._training_history

    def update_training_history(self, value: list) -> Parameters:
        return Parameters(
            self.params,
            self.bijectors,
            self.trainables,
            self.training_history.append(value),
        )

    def unpack(self):
        """Unpack the state into a tuple of parameters, trainables and bijectors.

        Returns:
            Tuple[Dict, Dict, Dict]: The parameters, trainables and bijectors.
        """
        return self.params, self.trainables, self.bijectors

    def constrain(self):
        return self.update_params(
            jtu.tree_map(
                lambda param, trans: trans.forward(param), self.params, self.bijectors
            )
        )

    def unconstrain(self):
        return self.update_params(
            jtu.tree_map(
                lambda param, trans: trans.inverse(param), self.params, self.bijectors
            )
        )

    def stop_gradients(self):
        def _stop_grad(param: Dict, trainable: Dict) -> Dict:
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, param)

        return self.update_params(
            jtu.tree_map(
                lambda param, trainable: _stop_grad(param, trainable),
                self.params,
                self.trainables,
            )
        )


__all__ = ["Parameters"]
