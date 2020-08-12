# TODO: use db like redis with TTL
from typing import List, Tuple
from eeval.server.utils import get_random_id
import tenseal as ts

DATASETS = {
    # dataset_id: [ctx_id, X, Y, batch_size]
}

CONTEXTS = {
    # ctx_id: context
}


def save_context(context: bytes) -> str:
    """Save a context into a permanent storage"""
    ctx_id = get_random_id()
    CONTEXTS[ctx_id] = context
    return ctx_id


def load_context(ctx_id: str) -> ts._ts_cpp.TenSEALContext:
    """Load a TenSEALContext"""
    context = get_raw_context(ctx_id)
    ctx = ts.context_from(context)
    return ctx


def get_raw_context(ctx_id: str) -> bytes:
    return CONTEXTS[ctx_id]


def save_dataset(ctx_id: str, X: List[bytes], Y: List[bytes], batch_size: int) -> str:
    """Save a dataset into a permanent storage"""
    dataset_id = get_random_id()
    DATASETS[dataset_id] = [ctx_id, X, Y, batch_size]
    return dataset_id


def load_dataset(
    dataset_id: str,
) -> Tuple[
    ts._ts_cpp.TenSEALContext,
    List[ts._ts_cpp.CKKSVector],
    List[ts._ts_cpp.CKKSVector],
    int,
]:
    """Load a dataset into CKKSVectors"""
    ctx_id, X, Y, batch_size = get_raw_dataset(dataset_id)
    ctx = load_context(ctx_id)
    enc_X = [ts.ckks_vector_from(ctx, x) for x in X]
    enc_Y = [ts.ckks_vector_from(ctx, y) for y in Y]
    return ctx, enc_X, enc_Y, batch_size


def get_raw_dataset(dataset_id: str) -> Tuple[str, List[bytes], List[bytes], int]:
    ctx_id, X, Y, batch_size = DATASETS[dataset_id]
    return ctx_id, X, Y, batch_size
