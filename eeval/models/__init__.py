"""Define models to process encrypted data"""
import os
import logging
import pickle
from typing import List
from collections import namedtuple
from eeval.models.fc import FC
from eeval.models.abstract_model import Model
from eeval.models.exceptions import ModelNotFound


_DATA_PATH = "data"

model_def = namedtuple(
    "model_definition", ["constructor", "default_version", "versions"]
)
# model names and version should always be lowercase
_MODEL_DEFS = {"fc": model_def(constructor=FC, default_version="0.1", versions=["0.1"])}
_MODELS = {
    model_name: {
        model_version: None for model_version in _MODEL_DEFS[model_name].versions
    }
    for model_name in _MODEL_DEFS.keys()
}


def _load_parameters(model_name: str, version: str) -> dict:
    """Load parameters for `model_name`:`version` from the appropriate file.

    Args:
        model_name: the name of the model
        version: the verion of the model

    Returns:
        dict: parameters loaded from the appropriate file

    Raises:
        OSError: if can't open parameters' file
    """
    file_name = f"{model_name}-{version}.parameters"
    file_path = os.path.join(
        os.path.realpath(os.path.dirname(__file__)), _DATA_PATH, file_name
    )
    try:
        parameters = pickle.load(open(file_path, "rb"))
    except OSError as ose:
        logging.error(
            f"Parameters file `{file_path}` not found for model `{model_name}`, version {version}"
        )
        raise ose
    return parameters


def get_model(model_name: str, version: str = None) -> Model:
    """Get model `model_name`:`version` if it exists.

    Args:
        model_name: the name of the model to use
        version: version of the model to use, a default version will be used if not specified

    Returns:
        Model: loaded model

    Raises:
        ModelNotFound: if the `model_name` or `version` are incorrect
    """

    # check if (model_name, version) is available
    if model_name not in _MODEL_DEFS.keys():
        raise ModelNotFound(f"Model `{model_name}` can't be found in this server")
    if version is None:
        version = _MODEL_DEFS[model_name].default_version
    elif version not in _MODEL_DEFS[model_name].versions:
        raise ModelNotFound(f"Model `{model_name}` doesn't have version `{version}`")

    # lazy loading of models
    if _MODELS[model_name][version] is None:
        parameters = _load_parameters(model_name, version)
        _MODELS[model_name][version] = _MODEL_DEFS[model_name].constructor(parameters)

    return _MODELS[model_name][version]


def get_model_def(model_name: str) -> dict:
    """Get descriptive attributes of model `model_name`.

    Args:
        model_name: the name of the model

    Returns:
        dict: containing descriptive model attributes

    Raises:
        ModelNotFound: if the model doesn't exist
    """
    try:
        model_def = _MODEL_DEFS[model_name]
    except KeyError:
        raise ModelNotFound(f"Model `{model_name}` can't be found in this server")

    return {
        "model_name": model_name,
        "description": model_def.constructor.__doc__,
        "default_version": model_def.default_version,
        "versions": model_def.versions,
    }


def get_all_model_def() -> List[dict]:
    """Get the description of all the available model"""
    model_defs = [get_model_def(model_name) for model_name in _MODEL_DEFS]
    return model_defs
