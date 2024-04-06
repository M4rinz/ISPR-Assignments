class DuplicateNodeIDError(Exception):
    """Raised when more than one node has the same ID."""
    pass

class InvalidArcException(Exception):
    """Raised if the arc trying to be inserted is invalid."""
    pass

class WrongAssignment(Exception):
    """
    Raised if the assignment for the conditioning side of the 
    probability is under/overcomplete. That is, if the row of 
    the table is not the length it should have
    """
    pass