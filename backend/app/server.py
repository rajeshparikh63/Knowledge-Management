from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.settings import settings
from app.middleware import  SecurityHeadersMiddleware
from app.logger import logger
from routers import health, upload, chat, models, auth, mindmap, report_suggestions, reports, flashcards, podcast, tak, simple_chat, google_drive, sharepoint, public_pages


def _prewarm_clients():
    """Pre-warm heavy clients in a background thread so first request is fast."""
    import threading

    def _warm():
        try:
            from services.ingestion_service import get_ingestion_service
            logger.info("🔥 Pre-warming IngestionService (GraphRAG, unstructured)...")
            _ = get_ingestion_service()
            logger.info("✅ IngestionService pre-warmed")

            from clients.kg.pipeline import get_kg_pipeline
            logger.info("🔥 Pre-warming KG pipeline...")
            _ = get_kg_pipeline()
            logger.info("✅ KG pipeline pre-warmed — first chat query will be fast")
        except Exception as e:
            logger.warning(f"⚠️ Pre-warm failed (non-fatal): {e}")

    threading.Thread(target=_warm, daemon=True).start()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifespan event handler for startup and shutdown events
    """
    # Startup
    logger.info("🚀 Starting SoldierIQ Backend...")
    logger.info(f"Environment: {settings.ENVIRONMENT}")

    # Pre-warm heavy clients in background so first dashboard load is fast
    _prewarm_clients()

    yield

    # Shutdown
    logger.info("🛑 Shutting down SoldierIQ Backend...")


# Initialize FastAPI app
app = FastAPI(
    title="SoldierIQ Backend",
    description="Tactical Intelligence Knowledge Management System",
    version="0.1.0",
    lifespan=lifespan
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure this for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=[
        "Content-Type",
        "Authorization",
        "Accept",
        "Accept-Language",
        "Content-Language",
        "Origin",
        "X-Timezone-Offset"
    ],
)

# Add custom middleware
app.add_middleware(SecurityHeadersMiddleware)


# JSON status endpoint (kept for uptime/health probes that expect JSON).
# The human-facing landing page is served at "/" by the public_pages router.
@app.get("/status")
async def status_json():
    return JSONResponse(
        content={
            "message": "SoldierIQ Backend is operational",
            "version": "0.1.0",
            "status": "online"
        },
        status_code=200
    )


# Public, unauthenticated pages (home / privacy / terms) at the domain root —
# required for Google OAuth verification. No /api prefix.
app.include_router(public_pages.router)

# Register routers with /api prefix
app.include_router(auth.router, prefix="/api")
app.include_router(health.router, prefix="/api")
app.include_router(upload.router, prefix="/api")
app.include_router(chat.router, prefix="/api")
app.include_router(simple_chat.router, prefix="/api")
app.include_router(models.router, prefix="/api")
app.include_router(mindmap.router, prefix="/api")
app.include_router(report_suggestions.router, prefix="/api")
app.include_router(reports.router, prefix="/api")
app.include_router(flashcards.router, prefix="/api")
app.include_router(podcast.router, prefix="/api")
app.include_router(tak.router, prefix="/api")
app.include_router(google_drive.router, prefix="/api")
app.include_router(sharepoint.router, prefix="/api")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info"
    )
