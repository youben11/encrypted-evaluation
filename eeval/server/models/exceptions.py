"""Custom exceptions. Basically just giving meaningful names."""


class ModelNotFound(Exception):
    """When a model can't be found"""

    pass


class EvaluationError(Exception):
    """When a problem happens during evaluation"""

    pass


class DeserializationError(Exception):
    """When context or encrypted input can't be deserialized"""

    pass


class InvalidContext(Exception):
    """When the context isn't appropriate for a specific model"""

    pass
