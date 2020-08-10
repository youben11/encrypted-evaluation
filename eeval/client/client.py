"""Client implementing the communication with the server"""

from typing import List, Union, Tuple
import requests
import tenseal as ts
from base64 import b64encode, b64decode
from eeval.client.exceptions import *


class Client:
    """Client to request server for evaluation"""

    def __init__(self, hostname: str, port: int):
        self._base_url = f"http://{hostname}:{port}"

    def ping(self) -> bool:
        """Make sure the API is up

        Returns:
            bool: True if the API is up, False otherwise
        """
        url = self._base_url + "/ping"
        try:
            response = requests.get(url)
        except:
            return False
        if response.status_code != 200:
            return False
        elif response.json() != {"message": "pong"}:
            return False

        return True

    def list_models(self) -> List[dict]:
        """List the models available in the API

        Returns:
            List[dict]: List of dictionaries, each one contains information about a specific model
                        hosted in the API

        Raises:
            ConnectionError: if a connection can't be established with the API
        """
        url = self._base_url + "/models/"
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError
        models = response.json()
        return models

    def model_info(self, model_name: str) -> dict:
        """Request information about a specific model `model_name`

        Args:
            model_name: the model name to request information about

        Returns:
            dict: information about the model

        Raises:
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """
        url = self._base_url + f"/models/{model_name}"
        try:
            response = requests.get(url)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        model_info = response.json()
        return model_info

    def evaluate(
        self,
        model_name: str,
        context: Union[ts._ts_cpp.TenSEALContext, bytes],  # serialized or not
        ckks_vector: Union[ts._ts_cpp.CKKSVector, bytes],  # serialized or not
    ) -> ts._ts_cpp.CKKSVector:
        """Evaluate model `model_name` on the encrypted input data `ckks_vector`

        Args:
            model_name: the model name to use for evaluation
            context: TenSEALContext containing keys needed for evaluation
            ckks_vector: encrypted input to feed the model with

        Returns:
            tenseal.CKKSVector: encrypted output of the evaluation

        Raises:
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/eval/{model_name}"

        # don't serialize if already
        if not isinstance(context, bytes):
            ser_ctx = context.serialize()
        else:
            ser_ctx = context
        if not isinstance(ckks_vector, bytes):
            ser_vec = ckks_vector.serialize()
        else:
            ser_vec = ckks_vector

        data = {
            "context": b64encode(ser_ctx).decode(),
            "ckks_vector": b64encode(ser_vec).decode(),
        }

        try:
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        ser_result = response.json()["ckks_vector"]
        result = ts.ckks_vector_from(context, b64decode(ser_result))
        return result

    def register_context(
        self, context: Union[ts._ts_cpp.TenSEALContext, bytes],  # serialized or not
    ) -> str:
        """Register a context in the server and get an id to refer to it

        Args:
            context: TenSEALContext to register

        Returns:
            str: id to use to refer to the registered context

        Raises:
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/contexts/register"

        # don't serialize if already
        if not isinstance(context, bytes):
            ser_ctx = context.serialize()
        else:
            ser_ctx = context

        data = {
            "context": b64encode(ser_ctx).decode(),
        }

        try:
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        return response.json()["context_id"]

    def get_context(self, ctx_id: str,) -> ts._ts_cpp.TenSEALContext:
        """Get a previously registered context using a context_id

        Args:
            ctx_id: id of a previously registered context

        Returns:
            TenSEALContext

        Raises:
            ConnectionError: if a connection can't be established with the API
            ResourceNotFound: if the context identified with `ctx_id` can't be found
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/contexts/"
        data = {"context_id": ctx_id}

        try:
            response = requests.get(url, params=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        ser_ctx = response.json()["context"]
        ctx = ts.context_from(b64decode(ser_ctx))

        return ctx

    def register_dataset(
        self,
        ckks_vectors: List[Union[ts._ts_cpp.CKKSVector, bytes]],  # serialized or not
        context: Union[ts._ts_cpp.TenSEALContext, bytes] = None,  # serialized or not
        context_id: str = None,
        batch_size: int = 1,
    ) -> Tuple[str, str]:
        """Register a dataset in the server and get a dataset_id to refer to it.

        Args:
            ckks_vectors: list containing entries of the dataset
            context: TenSEALContext containing keys needed for computing over the dataset.
                `context_id` need to be None if it is specified
            context_id: id of a previously registered context. `context` need to be None if it is
                specified
            batch_size: the number of entries a single ckks_vector contains

        Returns:
            Tuple[str, str]: IDs of both the registered context and dataset (context_id, dataset_id)

        Raises:
            TypeError: if batch_size isn't an int
            ValueError: if batch_size < 1
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/datasets/register"

        if not isinstance(batch_size, int):
            raise TypeError("batch_size need to be an int")
        if batch_size < 1:
            raise ValueError("batch_size need to be greater or equal than 1")

        if context is not None and context_id is not None:
            raise ValueError("context and context_id can't be both set")

        # register context if passed
        if context is not None:
            context_id = self.register_context(context)

        dataset = []
        for ckks_vector in ckks_vectors:
            if not isinstance(ckks_vector, bytes):
                ser_vec = ckks_vector.serialize()
            else:
                ser_vec = ckks_vector
            dataset.append(ser_vec)

        data = {
            "context_id": context_id,
            "ckks_vectors": [
                b64encode(ckks_vector).decode() for ckks_vector in dataset
            ],
            "batch_size": batch_size,
        }

        try:
            response = requests.post(url, json=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        dataset_id = response.json()["dataset_id"]
        return (context_id, dataset_id)

    def get_dataset(
        self, dataset_id: str, context: ts._ts_cpp.TenSEALContext
    ) -> Tuple[str, List[ts._ts_cpp.CKKSVector], int]:
        """Get a previously registered dataset using its id

        Args:
            dataset_id: id referring to the previously saved dataset
            context: TenSEALContext used to load the dataset

        Returns:
            (context_id, ckks_vectors, batch_size)

        Raises:
            ConnectionError: if a connection can't be established with the API
            ResourceNotFound: if the dataset identified with `dataset_id` can't be found
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + f"/datasets/"
        data = {"dataset_id": dataset_id}

        try:
            response = requests.get(url, params=data)
        except requests.exceptions.ConnectionError:
            raise ConnectionError

        if response.status_code != 200:
            Client._handle_error_response(response)

        resp_json = response.json()
        ctx_id = resp_json["context_id"]
        batch_size = resp_json["batch_size"]
        ckks_vectors = []
        for buff in resp_json["ckks_vectors"]:
            ckks_vector = ts.ckks_vector_from(context, b64decode(buff))
            ckks_vectors.append(ckks_vector)

        return ctx_id, ckks_vectors, batch_size

    @staticmethod
    def _handle_error_response(response: requests.Response):
        """Handle the responses that aren't a success (200)"""
        if response.status_code == 404:
            error_msg = response.json()["message"]
            raise ResourceNotFound(error_msg)
        elif response.status_code == 418:
            error_msg = response.json()["message"]
            raise Answer418(error_msg)
        elif response.status_code == 500:
            raise ServerError("Server error")
