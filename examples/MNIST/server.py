import tenseal as ts
import eeval.server as server
from eeval import models
from eeval.server.models.exceptions import (
    DeserializationError,
    InvalidContext,
)


class ConvMNIST(models.Model):
    """CNN for classifying MNIST data.
    Input should be an encoded 28x28 matrix representing the image.
    TenSEAL can be used for encoding `tenseal.im2col_encoding(ctx, input_matrix, 7, 7, 3)`
    The input should also be normalized with a mean=0.1307 and an std=0.3081 before encryption.
    """

    def __init__(self, parameters):
        self.conv1_weight = parameters["conv1_weight"]
        self.conv1_bias = parameters["conv1_bias"]
        self.fc1_weight = parameters["fc1_weight"]
        self.fc1_bias = parameters["fc1_bias"]
        self.fc2_weight = parameters["fc2_weight"]
        self.fc2_bias = parameters["fc2_bias"]
        self.windows_nb = parameters["windows_nb"]

    def forward(self, enc_x):
        # conv layer
        channels = []
        for kernel, bias in zip(self.conv1_weight, self.conv1_bias):
            y = enc_x.conv2d_im2col(kernel, self.windows_nb) + bias
            channels.append(y)
        out = ts.pack_vectors(channels)
        # squaring
        out.square_()
        # no need to flat
        # fc1 layer
        out.mm_(self.fc1_weight).add_plain_(self.fc1_bias)
        # squaring
        out.square_()
        # output layer
        out.mm_(self.fc2_weight).add_plain_(self.fc2_bias)
        return out

    @staticmethod
    def prepare_input(context, ckks_vector):
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


if __name__ == "__main__":
    server.register_model(ConvMNIST, versions=["0.1"], data_dir="parameters")
    server.start(host="localhost", port=8000)
