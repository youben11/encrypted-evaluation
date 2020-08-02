import pytest


@pytest.fixture(scope="module")
def base_url():
    return "http://localhost:8000"


@pytest.fixture(scope="module")
def client(base_url):
    pass


def test_ping(client):
    assert client.ping() == True
