class JobNotFound(ValueError):
    """Raised when a job id does not exist. Subclasses ValueError so existing
    `except ValueError` code keeps working; the API maps it to HTTP 404."""


class InvalidTransition(ValueError):
    """Raised when a state change is not allowed by the transition table."""


class TransitionConflict(InvalidTransition):
    """Raised when another caller changed the job first (atomic update matched no row)."""


class PipelineError(RuntimeError):
    """A pipeline failure with a machine-readable code and the stage it happened in.
    The code and stage are stored on the job so users can see why it failed."""

    def __init__(self, message: str, code: str = "pipeline_error", stage: str | None = None):
        super().__init__(message)
        self.code = code
        self.stage = stage
