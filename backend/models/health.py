from pydantic import BaseModel
from datetime import datetime


class HealthCheckResponse(BaseModel):
    """Health check response model"""
    status: str
    message: str
    version: str
    timestamp: datetime
    environment: str
