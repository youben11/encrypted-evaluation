"""FC model"""

import tenseal as ts
from models.abstract_model import Model


class FC(Model):
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
            raise RuntimeError(f"{e.__class__.__name__}: {str(e)}")
        return out

    @staticmethod
    def deserialize_input(context: bytes, ckks_vector: bytes) -> ts._ts_cpp.CKKSVector:
        # TODO: check parameters or size and raise RuntimeError when needed
        ctx = ts.context_from(context)
        # check parameters if good for model evaluation or raise an error
        enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        # maybe check size here
        return enc_x
