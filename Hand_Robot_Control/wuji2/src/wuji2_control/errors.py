"""Errors raised by communication and supervised control."""


class Wuji2Error(RuntimeError):
    """Base error for this control package."""


class FeedbackError(Wuji2Error):
    """Joint feedback is incomplete, invalid, or stale."""


class ControlError(Wuji2Error):
    """A control interlock or session lifecycle check failed."""


class CleanupError(ControlError):
    """Stopping, restoring settings, or closing resources failed."""
