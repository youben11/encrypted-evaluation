from math import log2, ceil


def next_pow2(n: int):
    return 2 ** ceil(log2(n))
