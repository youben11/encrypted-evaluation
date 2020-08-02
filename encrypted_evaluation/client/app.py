import tenseal as ts
from encrypted_evaluation.client import Client

ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [40, 21, 21, 21, 21, 21, 40])
ctx.global_scale = 2 ** 21
ctx.generate_galois_keys()

vec = ts.ckks_vector(ctx, [0.01] * 64)

client = Client("http://localhost:8000")
is_up = client.ping()
print(f"[+] API is {'up' if is_up else 'down'}")
print("[*] Sending context and encrypted vector for evaluation")
result = client.evaluate("fc", ctx, vec)
print(f"[+] Result: {result.decrypt()}")