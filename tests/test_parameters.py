from jaxutils.parameters import Parameters
import jax
import pytest
import jax.numpy as jnp
import distrax as dx


@pytest.mark.parametrize("jit_compile", [False, True])
def test_priors(jit_compile):
    # Vanilla test for case where every parameter has a defined prior
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    priors = {"a": dx.Normal(0.0, 1.0), "b": dx.Normal(0.0, 1.0)}

    params = Parameters(params=param_vals, priors=priors)
    if jit_compile:
        lpd = jax.jit(params.log_prior_density)()
    else:
        lpd = params.log_prior_density()
    assert pytest.approx(lpd, 0.00001) == -4.3378773
    assert isinstance(lpd, jax.Array)

    # Check fn. works for no priors
    priors = {"a": None, "b": None}
    params = Parameters(params=param_vals, priors=priors)
    if jit_compile:
        lpd = jax.jit(params.log_prior_density)()
    else:
        lpd = params.log_prior_density()
    assert pytest.approx(lpd, 0.00001) == 0.0
    assert isinstance(lpd, jax.Array)

    # Check the fn. works for nested structures with incomplete priors
    param_vals = {
        "a": jnp.array([1.0]),
        "b": {"a": jnp.array([10.0]), "b": jnp.array([3.0])},
    }
    priors = {"a": None, "b": {"a": dx.Normal(0, 1.0), "b": dx.Gamma(2.0, 2.0)}}
    params = Parameters(params=param_vals, priors=priors)
    if jit_compile:
        lpd = jax.jit(params.log_prior_density)()
    else:
        lpd = params.log_prior_density()
    assert pytest.approx(lpd, 0.00001) == -54.434032
    assert isinstance(lpd, jax.Array)

    # Check the prior initialising works - by default, there are no priors
    params = Parameters(param_vals)
    if jit_compile:
        lpd = jax.jit(params.log_prior_density)()
    else:
        lpd = params.log_prior_density()
    assert pytest.approx(lpd, 0.00001) == 0.0
    assert isinstance(lpd, jax.Array)
