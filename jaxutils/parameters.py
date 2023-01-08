from typing import Dict
from warnings import warn

import jax.random as jr
from jax.random import KeyArray
from jaxutils import PyTree


def merge_dictionaries(base_dict: Dict, in_dict: Dict) -> Dict:
    """
    This will return a complete dictionary based on the keys of the first
    matrix. If the same key should exist in the second matrix, then the
    key-value pair from the first dictionary will be overwritten. The purpose of
    this is that the base_dict will be a complete dictionary of values such that
    an incomplete second dictionary can be used to update specific key-value
    pairs.

    Args:
        base_dict (Dict): Complete dictionary of key-value pairs.
        in_dict (Dict): Subset of key-values pairs such that values from this
            dictionary will take precedent.

    Returns:
        Dict: A dictionary with the same keys as the base_dict, but with
            values from the in_dict.
    """
    for k, _ in base_dict.items():
        if k in in_dict.keys():
            base_dict[k] = in_dict[k]
    return base_dict


################################
# Base operations
################################
class ParameterState(PyTree):
    """
    The state of the model. This includes the parameter set, which parameters
    are to be trained and bijectors that allow parameters to be constrained and
    unconstrained.
    """

    def __init__(self, params: Dict, trainables: Dict, bijectors: Dict) -> None:
        self.params = params
        self.trainables = trainables
        self.bijectors = bijectors

    def unpack(self):
        """Unpack the state into a tuple of parameters, trainables and bijectors.

        Returns:
            Tuple[Dict, Dict, Dict]: The parameters, trainables and bijectors.
        """
        return self.params, self.trainables, self.bijectors


def initialise(model, key: KeyArray = None, **kwargs) -> ParameterState:
    """
    Initialise the stateful parameters of any GPJax object. This function also
    returns the trainability status of each parameter and set of bijectors that
    allow parameters to be constrained and unconstrained.

    Args:
        model: The GPJax object that is to be initialised.
        key (KeyArray, optional): The random key that is to be used for
            initialisation. Defaults to None.

    Returns:
        ParameterState: The state of the model. This includes the parameter
            set, which parameters are to be trained and bijectors that allow
            parameters to be constrained and unconstrained.
    """

    if key is None:
        warn(
            "No PRNGKey specified. Defaulting to seed 123.",
            UserWarning,
            stacklevel=2,
        )
        key = jr.PRNGKey(123)
    params = model._initialise_params(key)

    if kwargs:
        _validate_kwargs(kwargs, params)
        for k, v in kwargs.items():
            params[k] = merge_dictionaries(params[k], v)

    bijectors = build_bijectors(params)
    trainables = build_trainables(params)

    return ParameterState(
        params=params,
        trainables=trainables,
        bijectors=bijectors,
    )


def _validate_kwargs(kwargs, params):
    for k, v in kwargs.items():
        if k not in params.keys():
            raise ValueError(f"Parameter {k} is not a valid parameter.")


def recursive_items(d1: Dict, d2: Dict):
    """
    Recursive loop over pair of dictionaries whereby the value of a given key in
    either dictionary can be itself a dictionary.

    Args:
        d1 (_type_): _description_
        d2 (_type_): _description_

    Yields:
        _type_: _description_
    """
    for key, value in d1.items():
        if type(value) is dict:
            yield from recursive_items(value, d2[key])
        else:
            yield (key, value, d2[key])


def recursive_complete(d1: Dict, d2: Dict) -> Dict:
    """
    Recursive loop over pair of dictionaries whereby the value of a given key in
    either dictionary can be itself a dictionary. If the value of the key in the
    second dictionary is None, the value of the key in the first dictionary is
    used.

    Args:
        d1 (Dict): The reference dictionary.
        d2 (Dict): The potentially incomplete dictionary.

    Returns:
        Dict: A completed form of the second dictionary.
    """
    for key, value in d1.items():
        if type(value) is dict:
            if key in d2.keys():
                recursive_complete(value, d2[key])
        else:
            if key in d2.keys():
                d1[key] = d2[key]
    return d1


__all__ = [
    "ParameterState",
    "initialise",
    "recursive_items",
    "recursive_complete",
]
