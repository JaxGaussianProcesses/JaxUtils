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
from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float
from typing import Optional

from .pytree import PyTree

class Dataset(PyTree):
    """Dataset class."""

    #TODO: Consider HeterotopicDataset and IsotopicDataset abstractions.

    def __init__(
        self,
        X: Optional[Float[Array, "N D"]] = None,
        y: Optional[Float[Array, "N Q"]] = None,
    ) -> None:
        """
        Args:
            X(Float[Array, "N D"]]): Input data.
            y(Float[Array, "N Q"]]): Output data.

        Returns:
            Dataset: A dataset object.
        """

        _check_shape(X, y)
        self.X = X
        self.y = y

    def __repr__(self) -> str:
        return (
            f"- Number of datapoints: {self.X.shape[0]}\n- Dimension: {self.X.shape[1]}"
        )

    def is_supervised(self) -> bool:
        """Returns True if the dataset is supervised."""
        return self.X is not None and self.y is not None

    def is_unsupervised(self) -> bool:
        """Returns True if the dataset is unsupervised."""
        return self.X is None and self.y is not None


    def __add__(self, other: Dataset) -> Dataset:
        """Combines two datasets into one. The right-hand dataset is stacked beneath left."""
        x = jnp.concatenate((self.X, other.X))
        y = jnp.concatenate((self.y, other.y))

        return Dataset(X=x, y=y)

    @property
    def n(self) -> int:
        """The number of observations in the dataset."""
        return self.X.shape[0]

    @property
    def in_dim(self) -> int:
        """The dimension of the input data."""
        return self.X.shape[1]

    @property
    def out_dim(self) -> int:
        """The dimension of the output data."""
        return self.y.shape[1]
        

def verify_dataset(ds: Dataset) -> None:
    """Apply a series of checks to the dataset to ensure that downstream operations are safe."""
    assert ds.X.ndim == 2, (
        "2-dimensional training inputs are required. Current dimension:"
        f" {ds.X.ndim}."
    )
    if ds.y is not None:
        assert ds.y.ndim == 2, (
            "2-dimensional training outputs are required. Current dimension:"
            f" {ds.y.ndim}."
        )
        assert ds.X.shape[0] == ds.y.shape[0], (
            "Number of inputs must equal the number of outputs. \nCurrent"
            f" counts:\n- X: {ds.X.shape[0]}\n- y: {ds.y.shape[0]}"
        )


def _check_shape(X: Float[Array, "N D"], y: Float[Array, "N Q"]) -> None:
    """Checks that the shapes of X and y are compatible."""
    if X is not None and y is not None:
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of rows. Got X.shape={X.shape} and y.shape={y.shape}."
            )
        
    if X is not None and X.ndim != 2:
        raise ValueError(
            f"X must be a 2-dimensional array. Got X.ndim={X.ndim}."
        )

    if y is not None and y.ndim != 2:
        raise ValueError(
            f"y must be a 2-dimensional array. Got y.ndim={y.ndim}."
        )

__all__ = [
    "Dataset",
]
