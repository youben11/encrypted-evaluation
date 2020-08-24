import tenseal as ts
import torch
from eeval import Client
from PIL import Image
from torchvision import transforms


def create_ctx():
    poly_mod_degree = 8192
    coeff_mod_bit_sizes = [40, 21, 21, 21, 21, 21, 21, 40]
    ctx = ts.context(ts.SCHEME_TYPE.CKKS, poly_mod_degree, -1, coeff_mod_bit_sizes)
    ctx.global_scale = 2 ** 21
    ctx.generate_galois_keys()
    return ctx


def load_input():
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    img = Image.open("samples/img_100.jpg")
    return transform(img).view(28, 28).tolist()


def prepare_input(ctx, plain_input):
    enc_input, windows_nb = ts.im2col_encoding(ctx, plain_input, 7, 7, 3)
    assert windows_nb == 64
    return enc_input


def print_probs(output):
    probs = torch.softmax(torch.tensor(output), 0)
    label_max = torch.argmax(probs)
    print("Probabilities by label:")
    for label, prob in enumerate(probs):
        if label == label_max:
            print(f"Label={label}: {prob * 100 : .1f}% (max)")
        else:
            print(f"Label={label}: {prob * 100 : .1f}%")


if __name__ == "__main__":
    client = Client("localhost", 8000)
    ctx = create_ctx()
    plain_input = load_input()
    enc_input = prepare_input(ctx, plain_input)
    # TODO: context shouldn't contain secret-key
    enc_result = client.evaluate("ConvMNIST", ctx, enc_input)
    result = enc_result.decrypt()
    assert len(result) == 10
    print_probs(result)
