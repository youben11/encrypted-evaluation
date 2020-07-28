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
        out = enc_x.matmul(self.w1)
        out += self.b1
        out.square_()
        out @= self.w2
        out += self.b2
        out.square_()
        out @= self.w3
        out += self.b3
        return out

    @staticmethod
    def deserialize_input(context: bytes, ckks_vector: bytes) -> ts._ts_cpp.CKKSVector:
        ctx = ts.context_from(context)
        # check parameters if good for model evaluation or raise an error
        enc_x = ts.ckks_vector_from(ctx, ckks_vector)
        # maybe check size here
        return enc_x
