"""Custom exceptions. Basically just giving meaningful names."""


class Answer418(Exception):
    """Request specific error, the meaning mostly depends on the request"""

    pass


class ServerError(Exception):
    """When the server returns a status 500 response"""

    pass


class ResourceNotFound(Exception):
    """When a resource isn't found on the remote server"""

    pass
