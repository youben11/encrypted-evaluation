import tenseal as ts
import typer
from encrypted_evaluation.client import Client


app = typer.Typer()

# ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [40, 21, 21, 21, 21, 21, 40])
# ctx.global_scale = 2 ** 21
# ctx.generate_galois_keys()

# vec = ts.ckks_vector(ctx, [0.01] * 64)

# client = Client("http://localhost:8000")
# is_up = client.ping()
# print(f"[+] API is {'up' if is_up else 'down'}")
# print("[*] Sending context and encrypted vector for evaluation")
# result = client.evaluate("fc", ctx, vec)
# print(f"[+] Result: {result.decrypt()}")


@app.command()
def ping(url: str):
    client = Client(url)
    is_up = client.ping()
    if is_up:
        typer.secho("API is up", fg=typer.colors.GREEN)
    else:
        typer.secho("API is down", fg=typer.colors.RED)


@app.command()
def list_models(url: str):
    client = Client(url)
    models = client.list_models()
    print(models)


@app.command()
def model_info(url: str, model_name: str):
    client = Client(url)
    model_info = client.model_info(model_name)
    print(model_info)


if __name__ == "__main__":
    app()
