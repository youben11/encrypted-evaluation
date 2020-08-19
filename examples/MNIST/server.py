import eeval.server as server
from eeval import models


class ConvMNIST(models.Model):
    def __init__(self, parameters):
        pass

    def forward(self, enc_x):
        pass

    @staticmethod
    def prepare_input(ctx, enc_x):
        pass


if __name__ == "__main__":
    server.register_model(ConvMNIST, versions=["0.1"])
    server.start(host="localhost", port=8000)
