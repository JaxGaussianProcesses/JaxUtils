----
This project has now been incorporated into [GPJax](https://github.com/JaxGaussianProcesses/GPJax).
----
# [JaxUtils](https://github.com/JaxGaussianProcesses/JaxUtils)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/JaxGaussianProcesses/JaxUtils/tree/master.svg?style=svg)](https://dl.circleci.com/status-badge/redirect/gh/JaxGaussianProcesses/JaxUtils/tree/master)

`JaxUtils` provides utility functions for the [`JaxGaussianProcesses`]() ecosystem.</h2>

# Contents

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
