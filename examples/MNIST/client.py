from eeval import Client
import tenseal as ts


def create_ctx():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** 21
    ctx.generate_galois_keys()
    return ctx


def load_input():
    pass


def prepare_input(ctx, plain_input):
    pass


if __name__ == "__main__":
    client = Client("localhost", 8000)
    ctx = create_ctx()
    plain_input = load_input()
    enc_input = prepare_input(ctx, plain_input)
    # TODO: context shouldn't contain secret-key
    enc_result = client.evaluate("ConvMNIST", ctx, enc_input)
    result = enc_result.decrypt()
