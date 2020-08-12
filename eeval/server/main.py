"""RESTful API providing the main evaluation service"""

import uvicorn
from enum import Enum
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from eeval.server.models import get_model, get_all_model_def, get_model_def
from eeval.server.models.exceptions import *
from eeval.server import storage
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


class Context(BaseModel):
    context: str = Field(
        ...,
        description="Serialized TenSEALContext containing the keys needed for the evaluation",
    )


class CKKSVectorWithContext(CKKSVector, Context):
    pass


class Dataset(BaseModel):
    context_id: str = Field(..., description="id of the context used with this dataset")
    X: List[str] = Field(
        ..., description="Serialized CKKSVectors representing the data features"
    )
    Y: List[str] = Field(
        ..., description="Serialized CKKSVectors representing the data labels"
    )
    batch_size: int = Field(1, min=1, description="Number of entries per CKKSVector")


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


def answer_404(msg: str):
    return JSONResponse(
        status_code=404,
        content={"message": f"Resource not found. Server says '{msg}'"},
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


# TODO:
# - use data (files?) instead of json to not have the need to base64
# - optionally use a context_id. May serve performing multiple evaluations
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
        encrypted_x = model.prepare_input(context, ckks_vector)
        encrypted_out = model(encrypted_x)
    except (DeserializationError, EvaluationError, InvalidContext) as error:
        return answer_418(str(error))

    return {"ckks_vector": b64encode(encrypted_out.serialize())}


@app.get("/ping")
async def ping():
    """Used to check if the API is up"""
    return {"message": "pong"}


@app.post(
    "/contexts/register", response_description="id of the registered context",
)
async def register_context(data: Context):
    """Register a context and get a context_id to refer to it"""
    # decode data from client
    try:
        context = b64decode(data.context)
    except:
        return answer_418("bad base64 strings")

    # TODO: try except possible exceptions
    ctx_id = storage.save_context(context)
    return {"context_id": ctx_id}


@app.get(
    "/contexts/",
    response_model=Context,
    response_description="A previously registered context referenced by `context_id`",
)
async def get_context(context_id: str):
    """Get a previously registered context"""
    try:
        ctx = storage.get_raw_context(context_id)
    except KeyError:
        return answer_404(f"No context with id {context_id}")
    return {
        "context": b64encode(ctx),
    }


@app.post(
    "/datasets/register", response_description="id of the registered dataset",
)
async def register_dataset(data: Dataset):
    """Register a dataset and get a dataset_id to refer to it"""
    # decode data from client
    try:
        enc_X = []
        enc_Y = []
        for x, y in zip(data.X, data.Y):
            enc_X.append(b64decode(x))
            enc_Y.append(b64decode(y))
    except:
        return answer_418("bad base64 strings")

    # TODO: try except possible exceptions
    dataset_id = storage.save_dataset(data.context_id, enc_X, enc_Y, data.batch_size)
    return {"dataset_id": dataset_id}


@app.get(
    "/datasets/",
    response_model=Dataset,
    response_description="A previously registered dataset referenced by `dataset_id`",
)
async def get_dataset(dataset_id: str):
    """Get a previously registered dataset"""
    try:
        ctx_id, X, Y, batch_size = storage.get_raw_dataset(dataset_id)
    except KeyError:
        return answer_404(f"No dataset with id {dataset_id}")
    return {
        "context_id": ctx_id,
        "X": [b64encode(x) for x in X],
        "Y": [b64encode(y) for y in Y],
        "batch_size": batch_size,
    }


def start(host="127.0.0.1", port=8000):
    uvicorn.run(app, host="127.0.0.1", port=8000)


if __name__ == "__main__":
    start()
