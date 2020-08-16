"""Client implementing the communication with the server"""

from typing import List, Union, Tuple
import requests
import tenseal as ts
from base64 import b64encode, b64decode
from eeval.client.exceptions import *
from eeval.client import utils


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
        enc_X: List[Union[ts._ts_cpp.CKKSVector, bytes]],  # serialized or not
        enc_Y: List[Union[ts._ts_cpp.CKKSVector, bytes]],  # serialized or not
        context: Union[ts._ts_cpp.TenSEALContext, bytes] = None,  # serialized or not
        context_id: str = None,
        batch_size: int = 1,
    ) -> Tuple[str, str]:
        """Register a dataset in the server and get a dataset_id to refer to it.

        Args:
            enc_X: encrypted features
            enc_Y: encrypted labels
            context: TenSEALContext containing keys needed for computing over the dataset.
                `context_id` need to be None if it is specified
            context_id: id of a previously registered context. `context` need to be None if it is
                specified
            batch_size: the number of entries a single ckks_vector contains

        Returns:
            Tuple[str, str]: IDs of both the registered context and dataset (context_id, dataset_id)

        Raises:
            TypeError: if batch_size isn't an int
            ValueError: 
                - if batch_size < 1
                - if context or context_id are both set or not set
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
        elif context_id is None:
            raise ValueError("context or context_id need to be set")

        ser_X, ser_Y = [], []
        for x, y in zip(enc_X, enc_Y):
            if not isinstance(x, bytes):
                ser_x = x.serialize()
            else:
                ser_x = x
            if not isinstance(y, bytes):
                ser_y = y.serialize()
            else:
                ser_y = y
            ser_X.append(ser_x)
            ser_Y.append(ser_y)

        data = {
            "context_id": context_id,
            "X": [b64encode(x).decode() for x in ser_X],
            "Y": [b64encode(y).decode() for y in ser_Y],
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
            (context_id, X, Y, batch_size)

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
        enc_X, enc_Y = [], []
        for x_buff, y_buff in zip(resp_json["X"], resp_json["Y"]):
            x = ts.ckks_vector_from(context, b64decode(x_buff))
            y = ts.ckks_vector_from(context, b64decode(y_buff))
            enc_X.append(x)
            enc_Y.append(y)

        return ctx_id, enc_X, enc_Y, batch_size

    def train_logreg(
        self,
        context: ts._ts_cpp.TenSEALContext,
        X: List[List[float]],
        Y: List[int],
        weights: List[float],
        bias: float,
        epochs: int = 1,
    ) -> Tuple[List[float], float]:
        """Train a logistic regression model

        Args:
            context: context to use for encryption/decryption of the datasets and parameters, as
                well as remote computation
            X: data features of the training dataset
            Y: data labels of the training dataset
            weights: vector of weights of the logistic regression model
            bias: bias of the logistic regression model
            epochs: number of iteration over the dataset

        Returns:
            Tuple[List[float], float]: updated weights and bias

        Raises:
            TypeError: if epochs is not an int
            ValueError: 
                - if epochs is less than 1
                - if X and Y doesn't have the same size
                - if context doesn't hold a secret-key
            ConnectionError: if a connection can't be established with the API
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        if not isinstance(epochs, int):
            raise TypeError("epochs must be an int")
        if epochs < 1:
            raise ValueError("epochs must be greater or equal than 1")

        if len(X) != len(Y):
            raise ValueError(
                "number of data entries doesn't match the number of labels"
            )

        if not context.has_secret_key():
            raise ValueError("context doesn't hold a secret-key")

        n_features = len(X[0])
        assert (
            len(weights) == n_features
        ), "weights size doesn't match the number of features"

        # BATCHED client implementation
        # size_with_pad = utils.next_pow2(n_features)
        # padd = [0] * (size_with_pad - n_features)
        # slots = 4096  # TODO: ctx.max_slots()
        # # this should be an int since both numbers should be a power of two
        # batch_size = slots // size_with_pad
        # # we will throw entries that doesn't constitute a full batch
        # n_batch = len(X) // batch_size
        # enc_X, enc_Y = [], []
        # for i in range(n_batch):
        #     # TODO: make sure X[i * batch_size + j] is of len n_features
        #     # padd and batch
        #     x = [X[i * batch_size + j] + padd for j in range(batch_size)]
        #     y = [Y[i * batch_size + j] for j in range(batch_size)]
        #     assert len(x) == batch_size * size_with_pad
        #     enc_X.append(ts.ckks_vector(context, x))
        #     assert len(y) == batch_size
        #     enc_Y.append(ts.ckks_vector(context, y))

        enc_X, enc_Y = [], []
        batch_size = 1
        padd = []
        for x, y in zip(X, Y):
            enc_X.append(ts.ckks_vector(context, x))
            enc_Y.append(ts.ckks_vector(context, [y]))


        _, dataset_id = self.register_dataset(
            enc_X, enc_Y, context=context, batch_size=batch_size
        )

        for _ in range(epochs):
            # encrypt parameters
            enc_weights = ts.ckks_vector(context, weights + padd)
            enc_bias = ts.ckks_vector(context, bias)
            # train a single epoch
            ser_weights_update, ser_bias_update = self._train_logreg(
                context, enc_weights, enc_bias, dataset_id
            )
            # deserialize
            weights_update = ts.ckks_vector_from(context, b64decode(ser_weights_update))
            bias_update = ts.ckks_vector_from(context, b64decode(ser_bias_update))
            # update parameters
            weights = [w + u for w, u in zip(weights, weights_update.decrypt())]
            bias = [bias[0] + bias_update.decrypt()[0]]

        return weights, bias

    def _train_logreg(
        self,
        context: ts._ts_cpp.TenSEALContext,
        weights: ts._ts_cpp.CKKSVector,
        bias: ts._ts_cpp.CKKSVector,
        dataset_id: str,
    ) -> Tuple[ts._ts_cpp.CKKSVector, ts._ts_cpp.CKKSVector]:
        """Make a single pass through the remote encrypted dataset and get back encrypted
        parameter updates.

        Args:
            context: context used to deserialize parameters update
            weights: encrypted vector of weights
            bias: encrypted bias
            dataset_id: id of the remote dataset to train on

        Returns:
            Tuple[ts._ts_cpp.CKKSVector, ts._ts_cpp.CKKSVector]: weights and bias update

        Raises:
            ConnectionError: if a connection can't be established with the API
            ResourceNotFound: if the dataset identified with `dataset_id` can't be found
            Answer418: if response.status_code is 418
            ServerError: if response.status_code is 500
        """

        url = self._base_url + "/train-lr"

        data = {
            "weights": b64encode(weights.serialize()).decode(),
            "bias": b64encode(bias.serialize()).decode(),
            "dataset_id": dataset_id,
        }

        resp = requests.post(url, json=data)

        if resp.status_code != 200:
            Client._handle_error_response(resp)

        resp_json = resp.json()
        weights_update = resp_json["weights_update"]
        bias_update = resp_json["bias_update"]

        return weights_update, bias_update

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
        else:
            raise RuntimeError(f"Unkown server error -> [status_code: {response.status_code}]: '{response.text}'")
