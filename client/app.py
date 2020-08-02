import tenseal as ts
import requests
from base64 import b64encode, b64decode

ctx = ts.context(ts.SCHEME_TYPE.CKKS, 8192, -1, [40, 21, 21, 21, 21, 21, 40])
ctx.global_scale = 2 ** 21
ctx.generate_galois_keys()

vec = ts.ckks_vector(ctx, [0.01] * 64)

ser_ctx = ctx.serialize()
ser_vec = vec.serialize()

data = {
    "context": b64encode(ser_ctx).decode(),
    "ckks_vector": b64encode(ser_vec).decode(),
}

response = requests.post("http://127.0.0.1:8000/eval/fc", json=data)
status_code = response.status_code
print(f"status_code = {status_code}")

if status_code == 200:
    ser_result = response.json()["ckks_vector"]
    vec = ts.ckks_vector_from(ctx, b64decode(ser_result))
    print(vec.decrypt())
elif status_code == 418:
    print(f"message: {response.json()['message']}")
