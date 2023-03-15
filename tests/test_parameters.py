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
    priors = {k: dx.Normal(0.0, 1.0) for k in param_vals.keys()} if set_priors else None
    trainables = {k: True for k in param_vals.keys()} if set_trainables else None
    bijections = {k: Identity for k in param_vals.keys()} if set_bijectors else None
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

    unconstrain_fn = jax.jit(params.unconstrain) if jit_compile else params.unconstrain

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
    with pytest.raises(ValueError):
        updated_params = params.update_params({"a": Identity})


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
    with pytest.raises(ValueError):
        updated_params = params.update_priors({"a": dx.Gamma(3.0, 3.0)})


def param_equality(params, truth, set_priors, set_trainables, set_bijectors):
    assert params["params"] == truth["params"]

    if set_trainables:
        assert params["trainables"] == truth["trainables"]
    else:
        assert params["trainables"] == {k: True for k in truth["params"]}

    if set_bijectors:
        assert params["bijectors"] == truth["bijectors"]
    else:
        assert params["bijectors"] == {k: Identity for k in truth["params"]}

    if set_priors:
        assert params["priors"] == truth["priors"]
    else:
        assert params["priors"] == {k: None for k in truth["params"]}


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
    contents = params.unpack()

    param_equality(contents, truth, set_priors, set_trainables, set_bijectors)
    assert isinstance(contents["params"], dict)
    assert isinstance(contents["trainables"], dict)
    assert isinstance(contents["bijectors"], dict)
    assert isinstance(contents["priors"], dict)


@pytest.mark.parametrize("set_priors", [True, False])
@pytest.mark.parametrize("set_trainables", [True, False])
@pytest.mark.parametrize("set_bijectors", [True, False])
def test_combine(set_priors, set_trainables, set_bijectors):
    p1, truth1 = build_params(
        {"a": jnp.array([1.0])}, set_priors, set_trainables, set_bijectors
    )
    p2, truth2 = build_params(
        {"b": jnp.array([2.0])}, set_priors, set_trainables, set_bijectors
    )

    p = p1.combine(p2, left_key="x", right_key="y")
    assert isinstance(p, Parameters)
    assert p.params == {"x": truth1["params"], "y": truth2["params"]}
    assert list(p.keys()) == ["x", "y"]

    if set_trainables:
        assert p.trainables == {
            "x": truth1["trainables"],
            "y": truth2["trainables"],
        }
    else:
        assert p.trainables == {"x": {"a": True}, "y": {"b": True}}

    if set_bijectors:
        assert p.bijectors == {
            "x": truth1["bijectors"],
            "y": truth2["bijectors"],
        }
    else:
        assert p.bijectors == {"x": {"a": Identity}, "y": {"b": Identity}}

    if set_priors:
        assert p.priors == {"x": truth1["priors"], "y": truth2["priors"]}
    else:
        assert p.priors == {"x": {"a": None}, "y": {"b": None}}


@pytest.mark.parametrize("prior", [dx.Normal(0, 1), dx.Gamma(2.0, 2.0), None])
@pytest.mark.parametrize("trainable", [True, False])
@pytest.mark.parametrize("bijector", [Softplus, Identity])
def test_add_parameter(prior, trainable, bijector):
    p = Parameters({"a": jnp.array([1.0])})
    p.add_parameter(
        key="b",
        value=jnp.array([2.0]),
        prior=prior,
        trainability=trainable,
        bijector=bijector,
    )

    assert "b" in p.keys()
    assert p["b"] == jnp.array([2.0])
    assert p.trainables["b"] == trainable
    assert p.bijectors["b"] == bijector
    assert p.priors["b"] == prior

    # Test adding a parameter with a parameter object
    p = Parameters({"a": jnp.array([1.0])})
    p2 = Parameters(
        params={"c": jnp.array([2.0])},
        bijectors={"c": bijector},
        priors={"c": prior},
        trainables={"c": trainable},
    )
    p.add_parameter(
        key="b",
        parameter=p2,
    )

    assert "b" in p.keys()
    assert p["b"] == p2.params
    assert p.trainables["b"] == p2.trainables
    assert p.bijectors["b"] == p2.bijectors
    assert p.priors["b"] == p2.priors

    # Check that trying to overwrite a parameter raises an error
    with pytest.raises(ValueError):
        p.add_parameter(key="b", parameter=p2)
