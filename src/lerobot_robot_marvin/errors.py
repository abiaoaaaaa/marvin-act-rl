class MarvinError(RuntimeError):
    """Base deployment error."""


class ProtocolError(MarvinError):
    """Malformed or rejected RPC payload."""


class BridgeError(MarvinError):
    """Bridge returned an operation error."""


class SafetyError(MarvinError):
    """A command was rejected before it could reach HIL-SERL."""

