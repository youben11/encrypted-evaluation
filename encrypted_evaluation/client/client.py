from typing import List
import requests
import tenseal as ts
from base64 import b64encode, b64decode
from encrypted_evaluation.client.exceptions import *


class Client:
    """Client to request server for evaluation"""

    def __init__(self, base_url: str):
        self._base_url = base_url

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
        """
        url = self._base_url + "/models/"
        response = requests.get(url)
        models = response.json()
        return models

    def model_info(self, model_name: str) -> dict:
        """Request information about a specific model `model_name`

        Args:
            model_name: the model name to request information about

        Returns:
            dict: information about the model

        Raises:
            Exception: if response.status_code isn't 200
        """
        url = self._base_url + f"/models/{model_name}"
        response = requests.get(url)
        if response.status_code != 200:
            Client._handle_error_response(response)

        model_info = response.json()
        return model_info

    def evaluate(
        self,
        model_name: str,
        context: ts._ts_cpp.TenSEALContext,
        ckks_vector: ts._ts_cpp.CKKSVector,
    ) -> ts._ts_cpp.CKKSVector:
        """Evaluate model `model_name` on the encrypted input data `ckks_vector`

        Args:
            model_name: the model name to use for evaluation
            context: TenSEALContext containing keys needed for evaluation
            ckks_vector: encrypted input to feed the model with

        Returns:
            tenseal.CKKSVector: encrypted output of the evaluation

        Raises:
            Exception: if response.status_code isn't 200
        """

        url = self._base_url + f"/eval/{model_name}"

        ser_ctx = context.serialize()
        ser_vec = ckks_vector.serialize()
        data = {
            "context": b64encode(ser_ctx).decode(),
            "ckks_vector": b64encode(ser_vec).decode(),
        }

        response = requests.post(url, json=data)
        if response.status_code != 200:
            Client._handle_error_response(response)

        ser_result = response.json()["ckks_vector"]
        result = ts.ckks_vector_from(context, b64decode(ser_result))
        return result

    @staticmethod
    def _handle_error_response(response: requests.Response):
        """Handle the responses that aren't a success (200)"""
        # TODO: better exceptions
        if response.status_code == 418:
            error_msg = response.json()["message"]
            raise Answer418(error_msg)
        elif response.status_code == 500:
            raise ServerError("Server error")
