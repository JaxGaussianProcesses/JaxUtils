# [JaxUtils](https://github.com/JaxGaussianProcesses/JaxUtils)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/JaxGaussianProcesses/JaxUtils/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/JaxGaussianProcesses/JaxUtils/tree/master)

`JaxUtils` provides utility functions for the [`JaxGaussianProcesses`]() ecosystem.</h2>


## Training a Linear Model is easy peasy.
```python
import jaxutils as ju

from jaxutils.bijectors import Identity

import jax.numpy as jnp
import jax.random as jr
import optax as ox

# (1) Create a dataset:
X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
y = 2.0 * X + 1.0 + jr.normal(jr.PRNGKey(0), X.shape).reshape(-1, 1)
D = ju.Dataset(X, y)


# (2) Define your model:
class LinearModel(ju.Module):
    weight: float =  ju.param(Identity)
    bias: float = ju.param(Identity)

    def __call__(self, x):
        return self.weight * x + self.bias

model = LinearModel(weight=1.0, bias=1.0)


# (3) Define your loss function:
class MeanSqaureError(ju.Objective):

    def evaluate(self, model: LinearModel, train_data: ju.Dataset) -> float:
        return jnp.sum((train_data.y - model(train_data.X)) ** 2)

loss = MeanSqaureError()

# (4) Train!
model, hist = ju.fit(model, loss, D, ox.sgd(0.0001), 10000)

# (5) Check the results:
print(model.weight, model.bias)
```


# Contents - TO UPDATE.

- [PyTree](#pytree)
- [Dataset](#dataset)

# PyTree

## Overview

`jaxutils.PyTree` is a mixin class for [registering a python class as a JAX PyTree](https://jax.readthedocs.io/en/latest/pytrees.html#extending-pytrees). You would define your Python class as follows.

```python
class MyClass(jaxutils.PyTree):
    ...

```

## Example

```python
import jaxutils

from jaxtyping import Float, Array

class Line(jaxutils.PyTree):
    def __init__(self, gradient: Float[Array, "1"], intercept: Float[Array, "1"]) -> None
        self.gradient = gradient
        self.intercept = intercept

    def y(self, x: Float[Array, "N"]) -> Float[Array, "N"]
        return x * self.gradient + self.intercept
```

# Dataset

## Overview

`jaxutils.Dataset` is a datset abstraction. In future, we wish to extend this to a heterotopic and isotopic data abstraction.

## Example

```python
import jaxutils
import jax.numpy as jnp

# Inputs
X = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

# Outputs
y = jnp.array([[7.0], [8.0], [9.0]])

# Datset
D = jaxutils.Dataset(X=X, y=y)

print(f'The number of datapoints is {D.n}')
print(f'The input dimension is {D.in_dim}')
print(f'The output dimension is {D.out_dim}')
print(f'The input data is {D.X}')
print(f'The output data is {D.y}')
print(f'The data is supervised {D.is_supervised()}')
print(f'The data is unsupervised {D.is_unsupervised()}')
```

```
The number of datapoints is 3
The input dimension is 2
The output dimension is 1
The input data is [[1. 2.]
 [3. 4.]
 [5. 6.]]
The output data is [[7.]
 [8.]
 [9.]]
The data is supervised True
The data is unsupervised False
```

You can also add dataset together to concatenate them.

```python
# New inputs
X_new = jnp.array([[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]])

# New outputs
y_new = jnp.array([[7.0], [8.0], [9.0]])

# New dataset
D_new = jaxutils.Dataset(X=X_new, y=y_new)

# Concatenate the two datasets
D = D + D_new

print(f'The number of datapoints is {D.n}')
print(f'The input dimension is {D.in_dim}')
print(f'The output dimension is {D.out_dim}')
print(f'The input data is {D.X}')
print(f'The output data is {D.y}')
print(f'The data is supervised {D.is_supervised()}')
print(f'The data is unsupervised {D.is_unsupervised()}')
```

```
The number of datapoints is 6
The input dimension is 2
The output dimension is 1
The input data is [[1.  2. ]
 [3.  4. ]
 [5.  6. ]
 [1.5 2.5]
 [3.5 4.5]
 [5.5 6.5]]
The output data is [[7.]
 [8.]
 [9.]
 [7.]
 [8.]
 [9.]]
The data is supervised True
The data is unsupervised False
```
