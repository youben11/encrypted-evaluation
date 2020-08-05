"""FC model"""

import tenseal as ts
from eeval.server.models.abstract_model import Model
from eeval.server.models.exceptions import (
    DeserializationError,
    EvaluationError,
    InvalidContext,
)


class FC(Model):
    """Neural netowrk with 3 Linear layers followed by a square activation function,
    the last linear layer doesn't apply an activation function.

    input -> [LinearLayer -> square] -> [LinearLayer -> square] -> [LinearLayer] -> output

    The input should be encrypted as a tenseal.CKKSVector, the output will as well be encrypted.
    """

    def __init__(self, parameters):
        self.w1 = parameters["w1"]
        self.b1 = parameters["b1"]
        self.w2 = parameters["w2"]
        self.b2 = parameters["b2"]
        self.w3 = parameters["w3"]
        self.b3 = parameters["b3"]

    def forward(self, enc_x: ts._ts_cpp.CKKSVector) -> ts._ts_cpp.CKKSVector:
        try:
            out = enc_x.mm(self.w1) + self.b1
            out.square_()
            out = out.mm(self.w2) + self.b2
            out.square_()
            out = out.mm(self.w3) + self.b3
        except Exception as e:
            raise EvaluationError(f"{e.__class__.__name__}: {str(e)}")
        return out

    @staticmethod
    def deserialize_input(context: bytes, ckks_vector: bytes) -> ts._ts_cpp.CKKSVector:
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
