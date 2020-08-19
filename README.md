# encrypted-evaluation

A Python framework to build client/server applications, where the server provide services, such as model evaluation over encrypted inputs, then the client can encrypt his model input and send it to the server for evaluation and get back the encrypted result.


## Features

- :fire: Hosting models for encrypted evaluation over a RESTful API
- :cyclone: Client to send encrypted inputs for evaluation in a remote API
- :zap: CLI utility to encrypt/decrypt, generate keys, and communicate between the client and server

## Usage

You can use `eeval` either using the CLI or the programming API in Python.

### CLI

The CLI comes with the Python installation and it contains well documented instruction on how to make the job done. We expect it to be easy to use. If you find some difficulties using it, please open an issue to let us know how can it be better.

```bash
$ eeval
Usage: eeval [OPTIONS] COMMAND [ARGS]...

  Encrypted evaluation with homomorphic encryption

Options:
  -v, --verbose         verbose level  [default: 0]
  --install-completion  Install completion for the current shell.
  --show-completion     Show completion for the current shell, to copy it or
                        customize the installation.

  --help                Show this message and exit.

Commands:
  create-context  Create a TenSEAL context holding encryption keys and...
  decrypt         Decrypt a saved tensor
  encrypt         Encrypt a pickled numpy tensor
  eval            Evaluate an encrypted input on a remote hosted model
  list-models     List models available
  model-info      Get information about a specific model
  ping            Check if the API at URL is up
  serve           Start the API server
```

### API

We show a basic client/server app where the client send an encrypted input to the server to be evaluated over a linear layer (nothing really useful), more advanced usage can be found on our [examples section](#examples).


#### Server

Here we use the linear layer model which is already implemented for showcasing purposes, otherwise, you should implement your own model by inheriting from `eeval.server.model.Model` and implementing the required method

```python
import eeval.server as server
from eeval import models


# register the LinearLayer model
server.register_model(models.LinearLayer, versions=["0.1"])

server.start(host="localhost", port=8000)
```

#### Client

The only thing the client need to know is how to encode and encrypt his data, the rest is handled by `eeval.client.Client`

```python
from eeval import Client
import tenseal as ts


hostname = "localhost"
port = 8000
client = Client(hostname, port)

# prepare the TenSEAL context
ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [60, 40, 60])
ctx.global_scale = 2 ** 40
ctx.generate_galois_keys()
# we know that the model hosted on the server needs a vector of 16 as input
vec = [0.1] * 16
enc_vec = ts.ckks_vector(ctx, vec)

# print some info about the models hosted on the server
models = client.list_models()
print("============== Models ==============")
print("")
for i, model in enumerate(models):
    print(f"[{i + 1}] Model {model['model_name']}:")
    print(f"[*] Description: {model['description']}")
    print(f"[*] Versions: {model['versions']}")
    print(f"[*] Default version: {model['default_version']}")
    print("")

print("====================================")

# send the encrypted input and get encrypted output
result = client.evaluate(model_name="LinearLayer", context=ctx, ckks_vector=enc_vec)
print(f"decrypted result from the server: {result.decrypt()}")
```


## Installation

```bash
$ pip install eeval
```


## Examples

Example applications can be found under [examples](./examples).

## License

[GNU General Public License 3.0](https://github.com/youben11/encrypted-evaluation/blob/master/LICENSE)
