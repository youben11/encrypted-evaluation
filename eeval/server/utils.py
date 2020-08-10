from random import randint
from binascii import hexlify


TOKEN_LENGTH = 32


def get_random_id():
    rand_bytes = [randint(0, 255) for i in range(TOKEN_LENGTH)]
    return hexlify(bytes(rand_bytes)).decode()
