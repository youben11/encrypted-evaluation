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
    # TODO: actual training
    return weights, bias
