from typing import Tuple, List
from random import randint
from binascii import hexlify
import tenseal as ts


TOKEN_LENGTH = 32


def get_random_id():
    rand_bytes = [randint(0, 255) for i in range(TOKEN_LENGTH)]
    return hexlify(bytes(rand_bytes)).decode()


def load_lr(
    ctx: ts._ts_cpp.TenSEALContext, ser_weights: bytes, ser_bias: bytes
) -> Tuple[ts._ts_cpp.CKKSVector, ts._ts_cpp.CKKSVector]:
    weights = ts.ckks_vector_from(ctx, ser_weights)
    bias = ts.ckks_vector_from(ctx, ser_bias)
    return weights, bias


def train_lr(
    weights: ts._ts_cpp.CKKSVector,
    bias: ts._ts_cpp.CKKSVector,
    X: List[ts._ts_cpp.CKKSVector],
    Y: List[ts._ts_cpp.CKKSVector],
    batch_size: int,
) -> Tuple[ts._ts_cpp.CKKSVector, ts._ts_cpp.CKKSVector]:
    """Train an encrypted logsitic regression model on encrypted data

    Args:
        weights: encrypted weights of the LR model
        bias: encrypted bias of the LR model
        X: encrypted data features
        Y: encrypted data labels
        batch_size: the number of batched entries into a single CKKSVector

    Returns:
        Parameters update: weights_update and bias_update
    """
    assert batch_size == 1, "Currently supports batch_size=1 only"
    delta_weights = 0
    delta_bias = 0
    for x, y in zip(X, Y):
        ## FORWARD
        # linear layer
        out = x.dot(weights) + bias
        # sigmoid approximation
        out.polyval_([0.5, 0.197, 0, -0.004])

        ## BACKWARD
        out_minus_y = out - y
        delta_weights += x * out_minus_y
        delta_bias += out_minus_y

    # compute parameters update
    delta_weights *= -1 / len(X)  # + weights * 0.05 regularization
    delta_bias *= -1 / len(X)

    return delta_weights, delta_bias
