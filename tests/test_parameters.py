from jaxutils.parameters import Parameters
from jaxutils.bijectors import Softplus, Identity
import jax
import pytest
import jax.numpy as jnp
import distrax as dx
from jax.config import config
import typing as tp

config.update("jax_enable_x64", True)


def build_params(
    param_vals: tp.Dict,
    set_priors: bool,
    set_trainables: bool,
    set_bijectors: bool,
) -> tp.Tuple[Parameters, tp.Dict]:
    priors = (
        {"a": dx.Normal(0.0, 1.0), "b": dx.Normal(0.0, 1.0)}
        if set_priors
        else None
    )
    trainables = {"a": True, "b": True} if set_trainables else None
    bijections = {"a": Identity, "b": Identity} if set_bijectors else None
    params = Parameters(
        params=param_vals,
        priors=priors,
        bijectors=bijections,
        trainables=trainables,
    )
    truth = {
        "params": param_vals,
        "priors": priors,
        "trainables": trainables,
        "bijectors": bijections,
    }
    return params, truth


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


@pytest.mark.parametrize("jit_compile", [False, True])
def test_constrain_unconstrain(jit_compile):
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    bijections = {"a": Softplus, "b": Softplus}
    params = Parameters(params=param_vals, bijectors=bijections)

    unconstrain_fn = (
        jax.jit(params.unconstrain) if jit_compile else params.unconstrain
    )

    unconstrained_params = unconstrain_fn()

    assert isinstance(unconstrained_params, Parameters)
    assert isinstance(unconstrained_params.params, dict)

    constrain_fn = (
        jax.jit(unconstrained_params.constrain)
        if jit_compile
        else unconstrained_params.constrain
    )
    constrained_params = constrain_fn()
    assert isinstance(unconstrained_params, Parameters)
    assert isinstance(unconstrained_params.params, dict)

    assert constrained_params == params


def test_update_param():
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    bijections = {"a": Softplus, "b": Softplus}
    params = Parameters(params=param_vals, bijectors=bijections)

    updated_param_vals = {"a": jnp.array([2.0]), "b": jnp.array([3.0])}
    updated_params = params.update_params(updated_param_vals)

    # Check the updated params are correct
    assert updated_params.params == updated_param_vals
    # Check that nothing else has changed
    assert updated_params.bijectors == params.bijectors
    assert updated_params.priors == params.priors
    assert updated_params.trainables == params.trainables

    # Check that a key structure raises an error
    with pytest.raises(ValueError):
        updated_params = params.update_params({"a": jnp.array([2.0])})


def test_bijector_update():
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    bijections = {"a": Softplus, "b": Softplus}
    params = Parameters(params=param_vals, bijectors=bijections)

    updated_bijections = {"a": Softplus, "b": Identity}
    updated_params = params.update_bijectors(updated_bijections)

    # Check that bijections have been updated
    assert updated_params.bijectors == updated_bijections
    # Check all else is equal
    assert updated_params == params
    assert updated_params.trainables == params.trainables
    assert updated_params.priors == params.priors

    # Check that a key structure raises an error
    # with pytest.raises(ValueError):
    #     updated_params = params.update_params({"a": Identity})


def test_trainables_update():
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    trainables = {"a": True, "b": True}
    params = Parameters(params=param_vals, trainables=trainables)

    updated_trainables = {"a": True, "b": False}
    updated_params = params.update_trainables(updated_trainables)

    # Check that bijections have been updated
    assert updated_params.trainables == updated_trainables
    # Check all else is equal
    assert updated_params == params
    assert updated_params.bijectors == params.bijectors
    assert updated_params.priors == params.priors

    # Check that a key structure raises an error
    with pytest.raises(ValueError):
        updated_params = params.update_trainables({"a": True})


def test_priors_update():
    param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    priors = {"a": dx.Normal(0.0, 1.0), "b": dx.Normal(0.0, 1.0)}
    params = Parameters(params=param_vals, priors=priors)

    updated_priors = {"a": dx.Normal(0.0, 1.0), "b": dx.Gamma(3.0, 3.0)}
    updated_params = params.update_priors(updated_priors)

    # Check that bijections have been updated
    assert updated_params.priors == updated_priors
    # Check all else is equal
    assert updated_params == params
    assert updated_params.trainables == params.trainables
    assert updated_params.bijectors == params.bijectors

    # Check that a key structure raises an error
    # with pytest.raises(ValueError):
    #     updated_params = params.update_priors({"a": dx.Gamma(3.0, 3.0)})


@pytest.mark.parametrize("set_priors", [True, False])
@pytest.mark.parametrize("set_trainables", [True, False])
@pytest.mark.parametrize("set_bijectors", [True, False])
def test_unpack(set_priors, set_trainables, set_bijectors):
    init_param_vals = {"a": jnp.array([1.0]), "b": jnp.array([2.0])}
    params, truth = build_params(
        init_param_vals,
        set_priors,
        set_trainables,
        set_bijectors,
    )
    param_vals, trainables, bijectors = params.unpack()

    assert param_vals == truth["params"]

    if set_trainables:
        assert trainables == truth["trainables"]
    else:
        assert trainables == {"a": True, "b": True}

    if set_bijectors:
        assert bijectors == truth["bijectors"]
    else:
        assert bijectors == {"a": Identity, "b": Identity}

    assert isinstance(param_vals, dict)
    assert isinstance(trainables, dict)
    assert isinstance(bijectors, dict)
