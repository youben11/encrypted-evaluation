"""CLI for using eeval"""

import logging
import tenseal as ts
import typer
import pickle
import numpy as np
from typing import List, Tuple
from pathlib import Path
from eeval.client import Client
from eeval.client.exceptions import Answer418, ServerError
from eeval import server


VERBOSE = 0

app = typer.Typer()


def check_power_of_two(value: int) -> int:
    if value & (value - 1) != 0 or value <= 0:
        raise typer.BadParameter("Only powers of two greater than zero are allowed")
    return value


def couldnt_connect(hostname, port):
    typer.echo(f"Couldn't connect to '{hostname}:{port}'", err=True)
    raise typer.Exit(code=1)


def log(msg, verbosity=1):
    if verbosity <= VERBOSE:
        typer.echo(msg)


def load_ctx_and_input(
    context_file: typer.FileBinaryRead, input_file: typer.FileBinaryRead = None
) -> Tuple[ts._ts_cpp.TenSEALContext, ts._ts_cpp.CKKSVector]:
    try:
        ctx = ts.context_from(context_file.read())
    except Exception as e:
        typer.echo(f"Couldn't load context: {str(e)}", err=True)
        raise typer.Exit(code=1)

    # only load context
    if input_file is None:
        return ctx, None

    try:
        enc_input = ts.ckks_vector_from(ctx, input_file.read())
    except Exception as e:
        typer.echo(f"Couldn't load encrypted input: {str(e)}", err=True)
        raise typer.Exit(code=1)
    return ctx, enc_input


@app.command()
def ping(
    hostname: str = typer.Argument(
        ..., help="hostname of the server (e.g. 'myapi.com')"
    ),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="port"),
):
    """Check if the API at URL is up"""
    client = Client(hostname, port)
    is_up = client.ping()
    if is_up:
        typer.secho("API is up", fg=typer.colors.GREEN)
    else:
        typer.secho("API is down", fg=typer.colors.RED)


@app.command()
def list_models(
    hostname: str = typer.Argument(
        ..., help="hostname of the server (e.g. 'myapi.com')"
    ),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="port"),
    only_names: bool = typer.Option(
        False, "--only-names", "-n", help="show only the model names"
    ),
):
    """List models available"""
    client = Client(hostname, port)
    try:
        models = client.list_models()
    except ConnectionError:
        couldnt_connect(hostname, port)

    if len(models) == 0:
        typer.echo("No model available!")
        return

    header = "============== Models =============="
    footer = "===================================="
    if only_names:
        typer.echo(header)
        typer.echo("")
        for i, model in enumerate(models):
            typer.echo(f"[{i + 1}] Model {model['model_name']}")
        typer.echo("")
        typer.echo(footer)
    else:
        typer.echo(header)
        typer.echo("")
        for i, model in enumerate(models):
            typer.echo(f"[{i + 1}] Model {model['model_name']}:")
            typer.echo(f"[*] Description: {model['description']}")
            typer.echo(f"[*] Versions: {model['versions']}")
            typer.echo(f"[*] Default version: {model['default_version']}")
            typer.echo("")
        typer.echo(footer)


@app.command()
def model_info(
    hostname: str = typer.Argument(
        ..., help="hostname of the server (e.g. 'myapi.com')"
    ),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="port"),
    model_name: str = typer.Argument(...),
):
    """Get information about a specific model"""
    client = Client(hostname, port)
    try:
        model = client.model_info(model_name)
    except Answer418 as e:
        assert "can't be found" in str(e)
        typer.echo(f"Model `{model_name}` doesn't exist", err=True)
        raise typer.Exit(code=1)
    except ConnectionError:
        couldnt_connect(hostname, port)

    typer.echo(f"[+] Model {model['model_name']}:")
    typer.echo(f"[*] Description: {model['description']}")
    typer.echo(f"[*] Versions: {model['versions']}")
    typer.echo(f"[*] Default version: {model['default_version']}")


# TODO:
# - encode and encrypt here
# - display the decrypted output if there is a secret key
# - choose to do softmax, or choose the max label
# - writing the output should be optional
@app.command("eval")
def evaluate(
    hostname: str = typer.Argument(
        ..., help="hostname of the server (e.g. 'myapi.com')"
    ),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="port"),
    model_name: str = typer.Argument(..., help="model to use for evaluation"),
    context_file: typer.FileBinaryRead = typer.Argument(
        ..., envvar="TENSEAL_CONTEXT", help="file to load the TenSEAL context from"
    ),
    input_file: typer.FileBinaryRead = typer.Argument(
        ..., help="file to load the input from"
    ),
    output_file: typer.FileBinaryWrite = typer.Argument(
        ..., help="file to write the encrypted output to"
    ),
    send_secret_key: bool = typer.Option(
        False,
        "--sk/--no-sk",
        "-s/-S",
        help="send the secret key with the context (if it contains it)",
    ),
    decrypt_result: bool = typer.Option(
        False, "--decrypt", "-d", help="decrypt result",
    ),
):
    """Evaluate an encrypted input on a remote hosted model"""

    ctx, enc_input = load_ctx_and_input(context_file, input_file)

    ctx_holds_sk = ctx.is_private()
    sk = ctx.secret_key() if ctx_holds_sk else None
    if not send_secret_key:
        if ctx_holds_sk:
            ctx.make_context_public()
            log("dropped secret key from context")
        else:
            log("context doesn't hold a secret key, nothing to drop")

    client = Client(hostname, port)
    try:
        enc_out = client.evaluate(model_name, ctx, enc_input)
    except Answer418 as e:
        if "can't be found" in str(e):
            typer.echo(f"Model `{model_name}` doesn't exist", err=True)
        else:
            typer.echo(f"Error: {str(e)}", err=True)
        raise typer.Exit(code=1)
    except ConnectionError:
        couldnt_connect(hostname, port)
    except ServerError:
        typer.echo("Server side error", err=True)
        raise typer.Exit(code=1)

    out = None
    if decrypt_result:
        if ctx_holds_sk:
            out = enc_out.decrypt(sk)
            log(f"decrypted output: {out}")
        else:
            typer.echo(
                "Context doesn't hold a secret key, can't decrypt result", err=True
            )

    if out:  # decrypted
        pickle.dump(out, output_file)
        log("saved decrypted result to output file")
    else:
        output_file.write(enc_out.serialize())
        log("saved encrypted result to output file")


@app.command()
def decrypt(
    context_file: typer.FileBinaryRead = typer.Argument(
        ..., envvar="TENSEAL_CONTEXT", help="file to load the TenSEAL context from"
    ),
    input_file: typer.FileBinaryRead = typer.Argument(
        ..., help="file to load the tensor to decrypt from"
    ),
    output_file: typer.FileBinaryWrite = typer.Argument(
        ..., help="file to save the plain tensor to"
    ),
):
    """Decrypt a saved tensor"""

    ctx, enc_input = load_ctx_and_input(context_file, input_file)
    if not ctx.is_private():
        typer.echo("Context doesn't hold a secret key, can't decrypt tensor", err=True)
        raise typer.Exit(code=1)
    result = enc_input.decrypt()
    log(f"decryption completed, result is: {result}")
    pickle.dump(result, output_file)
    log("decrypted result saved to output file")


@app.command()
def encrypt(
    context_file: typer.FileBinaryRead = typer.Argument(
        ..., envvar="TENSEAL_CONTEXT", help="file to load the TenSEAL context from"
    ),
    input_file: typer.FileBinaryRead = typer.Argument(
        ..., help="file to load the numpy tensor to encrypt from"
    ),
    output_file: typer.FileBinaryWrite = typer.Argument(
        ..., help="file to save the encrypted tensor to"
    ),
    file_type: str = typer.Option(
        "", "--type", "-t", help="type of the file to encode"
    ),
    method: str = typer.Option("", "--method", "-m", help="encoding method to use"),
):
    """Encrypt a pickled numpy tensor"""

    ctx, _ = load_ctx_and_input(context_file, None)
    try:
        tensor = pickle.load(input_file)
    except Exception as e:
        typer.echo(f"Error while unpickling tensor: {str(e)}", err=True)
        raise typer.Exit(code=1)

    if not isinstance(tensor, np.ndarray):
        typer.echo("Can't encrypt other than numpy tensors", err=True)
        raise typer.Exit(code=1)

    dim = len(tensor.shape)
    if dim > 1:
        typer.echo("Can't encrypt numpy tensors of dim > 1", err=True)
        raise typer.Exit(code=1)

    assert dim == 1
    vec = tensor.tolist()
    log("tensor encoded")
    enc_vec = ts.ckks_vector(ctx, vec)
    log("tensor encrypted")
    output_file.write(enc_vec.serialize())
    log("saved encrypted tensor")


@app.command()
def create_context(
    output_file: typer.FileBinaryWrite = typer.Argument(
        ..., help="file to save the context to"
    ),
    poly_modulus_degree: int = typer.Argument(
        ..., help="polynomial modulus degree", callback=check_power_of_two
    ),
    coeff_mod_bit_sizes: List[int] = typer.Argument(
        ..., help="bit size of the coeffcients modulus"
    ),
    global_scale: float = typer.Option(
        ..., "--scale", min=1, help="scale to use by default for CKKS encoding",
    ),
    gen_galois_keys: bool = typer.Option(
        False, "--gk/--no-gk", "-g/-G", help="generate galois keys"
    ),
    gen_relin_keys: bool = typer.Option(
        True, "--rk/--no-rk", "-r/-R", help="generate relinearization keys"
    ),
    save_secret_key: bool = typer.Option(
        True, "--sk/--no-sk", "-s/-S", help="save the secret key into the context"
    ),
):
    """Create a TenSEAL context holding encryption keys and parameters"""

    ctx = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=poly_modulus_degree,
        coeff_mod_bit_sizes=coeff_mod_bit_sizes,
    )
    # set scale
    ctx.global_scale = global_scale
    log("context created")

    if gen_relin_keys:
        # relin keys is always generated
        pass
    if gen_galois_keys:
        ctx.generate_galois_keys()
        log("galois keys generated")
    if not save_secret_key:
        # drop secret-key
        ctx.make_context_public()
        log("secret key dropped")

    output_file.write(ctx.serialize())
    log("context saved successfully!")


@app.command("serve")
def start_server(
    loader_script: typer.FileText = typer.Option(
        None,
        "--loader-script",
        "-l",
        help="python script to register models in the API",
    ),
    data_dir: Path = typer.Option(
        None,
        "--data-dir",
        "-d",
        exists=True,
        help="default directory to look for model's data",
    ),
    host: str = typer.Option("localhost", "--host", "-h", help="host address"),
    port: int = typer.Option(8000, "--port", "-p", min=1, max=65535, help="port"),
):
    """Start the API server"""
    if data_dir is not None:
        if not data_dir.is_dir():
            raise typer.BadParameter("'--data-dir' / '-d' must be a directory.")
        path = server.models.set_default_data_dir(data_dir)
        log(f"default model's data directory set to {path}")

    if loader_script is None:
        typer.echo("No model to register. Use -l to register models")
    else:
        script = loader_script.read()
        try:
            exec(script)
        except Exception as e:
            typer.echo(f"Failed running loader script '{loader_script.name}': {e}")
            raise typer.Exit(code=1)

        # print registered models
        models = server.models.get_all_model_def()
        registered_models = ", ".join([model["model_name"] for model in models])
        typer.echo(f"Registered models: {registered_models}")
    # start serving the API
    typer.echo("Starting the server now...")
    server.start(host=host, port=port)


@app.callback()
def main(
    verbose: int = typer.Option(0, "--verbose", "-v", count=True, help="verbose level")
):
    """Encrypted evaluation with homomorphic encryption"""
    global VERBOSE
    VERBOSE = verbose


def run_cli():
    app()


if __name__ == "__main__":
    run_cli()
