class DuplicateNodeIDError(Exception):
    """Raised when more than one node has the same ID."""
    pass

class InvalidArcException(Exception):
    """Raised if the arc trying to be inserted is invalid."""
    pass