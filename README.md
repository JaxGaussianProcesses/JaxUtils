# [JaxUtils](https://github.com/JaxGaussianProcesses/JaxUtils)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/JaxGaussianProcesses/JaxUtils/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/JaxGaussianProcesses/JaxUtils/tree/master)

[`JaxUtils`](https://github.com/JaxGaussianProcesses/JaxUtils) is a lightweight library built on [`Equinox`](https://github.com/patrick-kidger/equinox) purposed to provide clean (and fast) model training functionality. This library also serves as a backend for the [`JaxGaussianProcesses`]() ecosystem.</h2>


# Contents

- [Overview](#overview)
- [Module] (#)
- [Dataset](#dataset)

# Overview

`JaxUtils` is designed....


## Linear Model example.

We fit a simple one-dimensional linear regression model with a `weight` and a `bias` parameter.

### (1) Dataset

```python
# Import dependancies.
import jaxutils as ju
import jax.numpy as jnp
import jax.random as jr
import optax as ox
import matplotlib.pyplot as plt

# Simulate labels.
key = jr.PRNGKey(42)
X = jnp.linspace(0.0, 10.0, 100).reshape(-1, 1)
y = 2.0 * X + 1.0 + jr.normal(key, X.shape)

# Create dataset object.
D = ju.Dataset(X, y)
```

### (2) Model

A model is defined through inheriting from the `JaxUtils`'s `Module` object. 
```python
class LinearModel(ju.Module):
    weight: float =  ju.param(ju.Identity)
    bias: float = ju.param(ju.Identity)

    def __call__(self, x):
        return self.weight * x + self.bias

model = LinearModel(weight=1.0, bias=1.0)
```
The parameters are marked via the `param` field, whose argument is the default `Bijector` transformation for mapping the parameters to the unconstrained space for optimisation. In this case both of our `weight` and `bias` parameters are defined on the reals, so we use the `Identity` transform. Just like in typicall `Equinox` code, we can (optionally) define a foward pass of the model through the `__call__` method.

### (3) Objective

We can define any objective function, such as the mean squared error, via inheriting from the `Objective` object as follows.
```python
class MeanSquaredError(ju.Objective):

    def evaluate(self, model: LinearModel, train_data: ju.Dataset) -> float:
        return jnp.mean((train_data.y - model(train_data.X)) ** 2)

loss = MeanSquaredError()
```

### (4) Train!

We are now ready to train our model. This can simply be done using the `fit` callable.
```python
# Optimisation loop.
model, hist = ju.fit(model=model, objective=loss, train_data=D, optim=optim, num_iters=1000)
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
