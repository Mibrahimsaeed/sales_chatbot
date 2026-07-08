class NotFoundError(Exception):
    """Raised by services when a lookup finds nothing — the API layer
    catches this and turns it into a 404."""
    def __init__(self, message: str = "Resource not found"):
        self.message = message
        super().__init__(message)


class UnsupportedMetricError(Exception):
    """Raised by leaderboard_service for any metric outside its allowed
    set — the API layer catches this and turns it into a 400."""
    def __init__(self, metric: str):
        self.message = f"Unsupported metric: {metric}"
        super().__init__(self.message)