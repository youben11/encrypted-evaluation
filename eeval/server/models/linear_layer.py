"""Linear Layer to compute `out = encrypted_input.matmul(weight) + bias`"""

import tenseal as ts
from eeval.server.models.abstract_model import Model
from eeval.server.models.exceptions import (
    DeserializationError,
    EvaluationError,
    InvalidContext,
)


class LinearLayer(Model):
    """Linear Layer computing `out = encrypted_input.matmul(weight) + bias`
    input and output shapes depends on the parameters weight and bias
    The input should be encrypted as a tenseal.CKKSVector, the output will as well be encrypted.
    """

    def __init__(self, parameters):
        # parameters is the unpickled version file
        self.weight = parameters["weight"]
        self.bias = parameters["bias"]

    def forward(self, enc_x: ts._ts_cpp.CKKSVector) -> ts._ts_cpp.CKKSVector:
        try:
            out = enc_x.mm(self.weight) + self.bias
        except Exception as e:
            raise EvaluationError(f"{e.__class__.__name__}: {str(e)}")
        return out

    @staticmethod
    def prepare_input(context: bytes, ckks_vector: bytes) -> ts._ts_cpp.CKKSVector:
        # TODO: check parameters or size and raise InvalidParameters when needed
        try:
            ctx = ts.context_from(context)
            enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        except:
            raise DeserializationError("cannot deserialize context or ckks_vector")

        # TODO: replace this with a more flexible check when introduced in the API
        try:
            _ = ctx.galois_keys()
        except:
            raise InvalidContext("the context doesn't hold galois keys")

        return enc_x
