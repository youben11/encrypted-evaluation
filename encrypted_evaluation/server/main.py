from enum import Enum
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from encrypted_evaluation.models import get_model, get_all_model_def, get_model_def
from encrypted_evaluation.models.exceptions import *
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


class ModelDescription(BaseModel):
    model_name: str = Field(
        ..., description="Name of the model. Used to query an evaluation"
    )
    description: str = Field(
        ...,
        description="The description of the model architecture, as well the input that should be passed to it",
    )
    default_version: str = Field(
        ..., description="The default version used during evaluation"
    )
    versions: List[str] = Field(..., description="Available versions of the model")


def answer_418(msg: str):
    return JSONResponse(
        status_code=418, content={"message": f"Oops! Server says '{msg}'"},
    )


@app.get("/models/", response_model=List[ModelDescription])
async def list_models():
    """List available models with their description"""
    return get_all_model_def()


@app.get("/models/{model_name}", response_model=ModelDescription)
async def describe_model(model_name: str):
    """Describe model `model_name`"""
    try:
        model_def = get_model_def(model_name)
    except ModelNotFound as mnf:
        return answer_418(str(mnf))
    return model_def


# TODO: use data (files?) instead of json to not have the need to base64
@app.post(
    "/eval/{model_name}",
    response_model=CKKSVector,
    response_description="encrypted output of the model",
)
async def evaluation(data: CKKSVectorWithContext, model_name: str, version: str = None):
    """
    Evaluate encrypted input data using the model `model_name` (optionally using a specific `version`)

    - **ckks_vector**: a serialized CKKSVector representing the input to the model
    - **context**: a serialized TenSEALContext containing the keys needed for the evaluation
    """

    # fetch model
    try:
        model = get_model(model_name, version)
    except ModelNotFound as mnf:
        return answer_418(str(mnf))
    except:
        raise HTTPException(status_code=500)

    # decode data from client
    try:
        context = b64decode(data.context)
        ckks_vector = b64decode(data.ckks_vector)
    except:
        return answer_418("bad base64 strings")

    # deserialize input and do the evaluation
    try:
        encrypted_x = model.deserialize_input(context, ckks_vector)
        encrypted_out = model(encrypted_x)
    except EvaluationError as ee:
        return answer_418(str(ee))
    except DeserializationError as de:
        return answer_418(str(de))

    return {"ckks_vector": b64encode(encrypted_out.serialize())}


@app.get("/ping")
async def ping():
    """Used to check if the API is up"""
    return {"message": "pong"}
