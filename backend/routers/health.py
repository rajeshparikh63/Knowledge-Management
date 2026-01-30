from datetime import datetime
from fastapi import APIRouter
from app.logger import logger
from app.settings import settings
from models.health import HealthCheckResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthCheckResponse)
@router.get("/healthz", response_model=HealthCheckResponse)
async def health_check() -> HealthCheckResponse:
    """
    Health check endpoint
    Returns the current status of the application
    """
    logger.info("Health check requested")

    return HealthCheckResponse(
        status="healthy",
        message="SoldierIQ Backend is operational and ready for tactical operations",
        version="0.1.0",
        timestamp=datetime.now(),
        environment=settings.ENVIRONMENT
    )
