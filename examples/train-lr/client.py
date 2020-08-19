from eeval import Client
import numpy as np
import tenseal as ts
import random


random.seed(73)
np.random.seed(73)


def lr_accuracy(weights, bias, x_test, y_test):
    correct = 0
    for x, y in zip(x_test, y_test):
        out = np.dot(weights, x) + np.array(bias)
        out = 1 / (1 + np.exp(-out))
        if np.abs(out - y) < 0.5:
            correct += 1
    print(f"Accuracy: {correct}/{len(x_test)} = {correct / len(x_test)}")
    return correct / len(x_test)


def random_data(m=128, n=2):
    # data separable by the line `y = x`
    x_train = np.random.randn(m, n)
    x_test = np.random.randn(m // 2, n)
    y_train = x_train[:, 0] >= x_train[:, 1]
    y_test = x_test[:, 0] >= x_test[:, 1]
    return x_train.tolist(), y_train.tolist(), x_test.tolist(), y_test.tolist()


def create_ctx():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** 21
    ctx.generate_galois_keys()
    return ctx


if __name__ == "__main__":
    client = Client("localhost", 8000)
    # generate data
    m, n = 128, 2
    x_train, y_train, x_test, y_test = random_data(m, n)
    # create the LR model
    weights = np.random.randn(n).tolist()
    bias = np.random.randn(1).tolist()
    # accuracy before training
    lr_accuracy(weights, bias, x_test, y_test)
    # encrypted training
    ctx = create_ctx()
    weights, bias = client.train_logreg(ctx, x_train, y_train, weights, bias, epochs=5)
    # accuracy after training
    lr_accuracy(weights, bias, x_test, y_test)
