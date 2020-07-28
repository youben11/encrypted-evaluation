from enum import Enum
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from models import get_model
from base64 import b64encode, b64decode


app = FastAPI()

# middleware to record process time
# from fastapi import Request
# @app.middleware("http")
# async def add_process_time_header(request: Request, call_next):
#     start_time = time.time()
#     response = await call_next(request)
#     process_time = time.time() - start_time
#     response.headers["X-Process-Time"] = str(process_time)
#     return response

# enable any origin for CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=False,
#     allow_methods=["POST"],
#     allow_headers=["*"],
# )


class ModelName(str, Enum):
    conv = "conv"
    fc = "fc"


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
    data: CKKSVectorWithContext, model_name: ModelName, version: str = None
):
    """
    Evaluate encrypted input data using the model `model_name` (optionally using a specific `version`)

    - **ckks_vector**: a serialized CKKSVector representing the input to the model
    - **context**: a serialized TenSEALContext containing the keys needed for the evaluation
    """

    try:
        model = get_model(model_name, version)
    except KeyError as ke:
        # bad input no model with this name or version
        return JSONResponse(
            status_code=418, content={"message": f"Oops! Server says '{ke.__str__()}'"},
        )

    try:
        context = b64decode(data.context)
        ckks_vector = b64decode(data.ckks_vector)
        encrypted_x = model.deserialize_input(context, ckks_vector)
        encrypted_out = model(encrypted_x)
    # models should raise RuntimeError when something is wrong about
    # deserialization as well
    except RuntimeError as re:
        # raise HTTPException(status_code=400, detail=re.__str__())
        # handle it with a beautiful error message
        return JSONResponse(
            status_code=418, content={"message": f"Oops! Server says '{re.__str__()}'"},
        )

    return {
        "out": b64encode(encrypted_out.serialize()).decode(),
    }


# TODO:
# - route /models/: list available models with reasonable description
# - route /model/{model_name}/: describe deeply the model with listing of available versions
