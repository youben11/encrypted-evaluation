# TODO: use db like redis with TTL
from typing import List, Tuple
from eeval.server.utils import get_random_id
import tenseal as ts

DATASETS = {
    # dataset_id: [ctx_id, dataset, batch_size]
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
    context = CONTEXTS[ctx_id]
    ctx = ts.context_from(context)
    return ctx


def get_raw_context(ctx_id: str) -> bytes:
    return CONTEXTS[ctx_id]


def save_dataset(ctx_id: str, dataset: List[bytes], batch_size: int) -> str:
    """Save a dataset into a permanent storage"""
    dataset_id = get_random_id()
    DATASETS[dataset_id] = [ctx_id, dataset, batch_size]
    return dataset_id


def load_dataset(dataset_id: str) -> Tuple[List[ts._ts_cpp.CKKSVector], int]:
    """Load a dataset into CKKSVectors"""
    ctx_id, dataset, batch_size = DATASETS[dataset_id]
    context = CONTEXTS[ctx_id]
    ctx = ts.context_from(context)
    ckks_vectors = [ts.ckks_vector_from(ctx, buff) for buff in dataset]
    return ckks_vectors, batch_size


def get_raw_dataset(dataset_id: str) -> Tuple[str, bytes, int]:
    ctx_id, dataset, batch_size = DATASETS[dataset_id]
    return ctx_id, dataset, batch_size
