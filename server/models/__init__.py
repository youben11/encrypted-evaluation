"""Define models to process encrypted data"""
import os
import pickle
from collections import namedtuple
from models.fc import FC
from models.abstract_model import Model
from models.exceptions import ModelNotFound


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
        # TODO: log this information
        # raise RuntimeError(f"Internal error: {file_path} should exist!")
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
