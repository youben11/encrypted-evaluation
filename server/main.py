from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from models import get_model
from models.exceptions import *
from base64 import b64encode, b64decode


BENCHMARK = True
CORS = True


app = FastAPI()

if BENCHMARK:
    from fastapi import Request
    import time

    @app.middleware("http")
    async def add_process_time_header(request: Request, call_next):
        tick = time.time()
        response = await call_next(request)
        tock = time.time()
        process_time = tock - tick
        print(f"Calling {request.url} took {process_time} seconds")
        response.headers["X-Process-Time"] = str(process_time)
        return response


if CORS:
    from fastapi.middleware.cors import CORSMiddleware

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["POST"],
        allow_headers=["*"],
    )


class CKKSVector(BaseModel):
    ckks_vector: str = Field(
        ..., description="Serialized CKKSVector representing the input to the model"
    )


class CKKSVectorWithContext(CKKSVector):
    context: str = Field(
        ...,
        description="Serialized TenSEALContext containing the keys needed for the evaluation",
    )


# @app.post("/eval/{model_name}", response_model=CKKSVector)
@app.post("/eval/{model_name}", response_description="encrypted output of the model")
async def evaluation(
    data: CKKSVectorWithContext, model_name: str, version: str = None
):
    """
    Evaluate encrypted input data using the model `model_name` (optionally using a specific `version`)

    - **ckks_vector**: a serialized CKKSVector representing the input to the model
    - **context**: a serialized TenSEALContext containing the keys needed for the evaluation
    """

    # fetch model
    try:
        model = get_model(model_name, version)
    except ModelNotFound as mnf:
        return JSONResponse(
            status_code=418, content={"message": f"Oops! Server says '{str(mnf)}'"},
        )
    except:
        raise HTTPException(status_code=500)

    # decode data from client
    try:
        context = b64decode(data.context)
        ckks_vector = b64decode(data.ckks_vector)
    except:
        return JSONResponse(
            status_code=418,
            content={"message": f"Oops! Server says 'bad base64 strings'"},
        )

    # deserialize input and do the evaluation
    try:
        encrypted_x = model.deserialize_input(context, ckks_vector)
        encrypted_out = model(encrypted_x)
    except EvaluationError as ee:
        return JSONResponse(
            status_code=418, content={"message": f"Oops! Server says '{str(ee)}'"},
        )
    except DeserializationError as de:
        return JSONResponse(
            status_code=418, content={"message": f"Oops! Server says '{str(de)}'"},
        )

    return {
        "out": b64encode(encrypted_out.serialize()).decode(),
    }


# TODO:
# - route /models/: list available models with reasonable description
# - route /model/{model_name}/: describe deeply the model with listing of available versions
