"""Server for hosting machine learning models to evaluate encrypted inputs"""

from eeval.server.main import start
from eeval.server.models import register_model


__all__ = ["start"]
