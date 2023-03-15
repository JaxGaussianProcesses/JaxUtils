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
import jax.numpy as jnp
from typing import Any, Callable, Dict
from .bijectors import Identity
from simple_pytree import Pytree, static_field
from jax.tree_util import tree_flatten, tree_structure
from jaxtyping import Float, Array
from distrax import Bijector, Distribution
from collections.abc import KeysView, ValuesView, ItemsView


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
        # TODO: Should block inplace updates e.g., `params.params['a']=jnp.array([2.])`
        params: Dict,
        bijectors: Dict = None,
        trainables: Dict = None,
        priors: Dict = None,
        training_history=None,
    ):

        if bijectors is None:
            bijectors = jtu.tree_map(lambda _: Identity, params)

        if trainables is None:
            trainables = jtu.tree_map(lambda _: True, params)

        if priors is None:
            priors = jtu.tree_map(lambda _: None, params)

        self._param_dict = params
        self._trainable_dict = trainables
        self._bijector_dict = bijectors
        self._prior_dict = priors
        self._training_history = training_history

    def __repr__(self) -> str:
        return f"Parameters({self.params.__repr__()})"

    def __getitem__(self, __name: str) -> Any:
        return self._param_dict.__getitem__(__name)

    def __setitem__(self, __name: str, __value: Any) -> None:
        return self._param_dict.__setitem__(__name, __value)

    def __eq__(self, other: Parameters) -> bool:
        return self.params == other.params

    @property
    def params(self) -> Dict:
        return self._param_dict

    def update_params(self, value: Dict) -> Parameters:
        self._validate_update(value, self.params, "params")
        return Parameters(
            value,
            self.bijectors,
            self.trainables,
            self.priors,
            self.training_history,
        )

    @staticmethod
    def _validate_update(
        value: dict,
        comparison: dict,
        name: str,
        lambda_expression: Callable[[Any], bool] = None,
    ):
        if tree_structure(comparison, lambda_expression) != tree_structure(
            value, lambda_expression
        ):
            raise ValueError(
                f"The structure of the {name} has changed. Please ensure"
                f" updates to {name} do not alter the strcuture."
            )

    @property
    def bijectors(self) -> Dict:
        return self._bijector_dict

    def update_bijectors(self, value: Dict) -> Parameters:
        self._validate_update(
            value,
            self.bijectors,
            "bijectors",
            lambda x: isinstance(x, Bijector),
        )
        return Parameters(
            self.params,
            value,
            self.trainables,
            self.priors,
            self.training_history,
        )

    @property
    def trainables(self) -> Dict:
        return self._trainable_dict

    def update_trainables(self, value: Dict) -> Parameters:
        self._validate_update(value, self.trainables, "trainables")
        return Parameters(
            self.params,
            self.bijectors,
            value,
            self.priors,
            self.training_history,
        )

    @property
    def priors(self) -> Dict:
        return self._prior_dict

    def update_priors(self, value: Dict) -> Parameters:
        self._validate_update(
            value,
            self.priors,
            "priors",
            lambda x: isinstance(x, Distribution),
        )
        return Parameters(
            self.params,
            self.bijectors,
            self.trainables,
            value,
            self.training_history,
        )

    @property
    def training_history(self) -> list:
        return self._training_history

    def update_training_history(self, value: list) -> Parameters:
        return Parameters(
            self.params,
            self.bijectors,
            self.trainables,
            self.priors,
            value,
        )

    def unpack(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Unpack the state into a tuple of parameters, trainables and bijectors.

        Returns:
            Dict[str, Dict[str, Any]]: The parameters, trainables and bijectors.
        """
        contents = {
            "params": self.params,
            "trainables": self.trainables,
            "bijectors": self.bijectors,
            "priors": self.priors,
        }
        return contents

    def constrain(self) -> Parameters:
        """Use the bijectors to transform the parameters to a constrained space.

        Returns:
            Parameters: A new Parameters object with the constrained parameter values.
        """
        return self.update_params(
            jtu.tree_map(
                lambda param, trans: trans.forward(param),
                self.params,
                self.bijectors,
            )
        )

    def unconstrain(self) -> Parameters:
        """Use the bijectors to transform the parameters to an unconstrained space.

        Returns:
            Parameters: A new Parameters object with the unconstrained parameter
                values.
        """
        return self.update_params(
            jtu.tree_map(
                lambda param, trans: trans.inverse(param),
                self.params,
                self.bijectors,
            )
        )

    def add_parameter(
        self,
        key: str,
        *,
        parameter: Parameters = None,
        value: jax.Array = None,
        prior: Distribution = None,
        bijector: Bijector = Identity,
        trainability: bool = True,
    ) -> None:
        """Add a parameter to the Parameters object.

        Args:
            key (str): The name of the parameter.
            value (jax.Array): The value of the parameter.
            prior (Distribution): The prior distribution of the parameter.
            bijector (Bijector): The bijector to transform the parameter.
            trainability (bool): The trainability of the parameter.
        """
        if key in self.keys():
            raise ValueError(f"Parameter with key: {key} already exists.")
        else:
            if parameter is None:
                self.params[key] = value
                self.priors[key] = prior
                self.bijectors[key] = bijector
                self.trainables[key] = trainability
            else:
                contents = parameter.unpack()
                self.params[key] = contents["params"]
                self.priors[key] = contents["priors"]
                self.bijectors[key] = contents["bijectors"]
                self.trainables[key] = contents["trainables"]

    def stop_gradients(self) -> Parameters:
        def _stop_grad(param: Dict, trainable: Dict) -> Dict:
            return jax.lax.cond(trainable, lambda x: x, jax.lax.stop_gradient, param)

        return self.update_params(
            jtu.tree_map(
                lambda param, trainable: _stop_grad(param, trainable),
                self.params,
                self.trainables,
            )
        )

    def items(self) -> ItemsView:
        """Return the items of the parameters."""
        return self.params.items()

    def keys(self) -> KeysView:
        """Return the keys of the parameters."""
        return self.params.keys()

    def values(self) -> ValuesView:
        """Return the values of the parameters."""
        return self.params.values()

    def log_prior_density(self) -> Array[Float, "1"]:
        """
        Recursive loop over pair of dictionaries that correspond to a parameter's
        current value and the parameter's respective prior distribution. For
        parameters where a prior distribution is specified, the log-prior density is
        evaluated at the parameter's current value.

        Args: params (Dict): Dictionary containing the current set of parameter
            estimates. priors (Dict): Dictionary specifying the parameters' prior
            distributions.

        Returns:
            Dict: The log-prior density, summed over all parameters.
        """

        def log_density(param, prior):
            # TODO: Use a jax.lax.cond be used here?
            if prior is not None:
                return jnp.sum(prior.log_prob(param))
            else:
                return jnp.array(0.0)

        log_prior_density_dict = jtu.tree_map(log_density, self.params, self.priors)
        leaves, _ = tree_flatten(log_prior_density_dict)
        return sum(leaves)

    def combine(self, other: Parameters, left_key: str, right_key: str) -> Parameters:
        """Combine two sets of parameters into a single set of parameters.

        Args:
            other (Parameters): The other set of parameters.
            left_key (str): The key to use for the left (i.e., `self`) set of
                parameters.
            right_key (str): The key to use for the right (i.e., `other`) set of
                parameters

        Returns:
            Parameters: A nested set of parameters.
        """

        self_contents = self.unpack()
        other_contents = other.unpack()

        combined_contents = {}
        for k in self_contents.keys():
            combined_contents[k] = {
                left_key: self_contents[k],
                right_key: other_contents[k],
            }

        return Parameters(
            params=combined_contents["params"],
            bijectors=combined_contents["bijectors"],
            trainables=combined_contents["trainables"],
            priors=combined_contents["priors"],
        )


__all__ = ["Parameters"]
