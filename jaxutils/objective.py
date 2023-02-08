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

import abc
import equinox as eqx

from .data import Dataset
from .module import Module

class Objective(eqx.Module):
    """Base class for objective functions."""
    negative: bool = eqx.static_field()
    constant: float = eqx.static_field()

    def __init__(self, negative: bool = False):
        """Initialise the objective function.

        Args:
            negative(bool): Whether to negate the objective function.
        
        Returns:
            Objective: An objective function.
        """
        
        self.negative = negative
        self.constant = -1.0 if negative else 1.0

    def __call__(self, model: Module, train_data: Dataset) -> float:
        """Evaluate the objective function.

        Args:
            model(Base): A model.
            *args: Additional arguments.
            **kwargs: Additional keyword arguments.
        
        Returns:
            float: The objective function.
        """
        return self.constant * self.evaluate(model, train_data)

    @abc.abstractmethod
    def evaluate(self, model: Module, train_data: Dataset) -> float:
        """Evaluate the objective function."""
        raise NotImplementedError


__all__ = [
    "Objective",
]
