import sys
import os
from pathlib import Path
from contextlib import asynccontextmanager
import logging

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Initialize config and thread safeguards first
from backend.config import settings

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError

from backend.deps import init_app_state, state
from backend.api.routes import api_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("cerebrum.main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle: Pre-loads models and static datasets once at startup."""
    logger.info("Starting SanDisk Cerebrum FastAPI application...")
    init_app_state()
    yield
    logger.info("Shutting down SanDisk Cerebrum FastAPI application.")

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description=(
        "Enterprise Semiconductor Die Yield Intelligence Platform API.\n\n"
        "Provides high-throughput screening (Model A), sub-die resolution inspection (Model B), "
        "and selective uncertainty-gated cascade inference across 27,972 canonical holdout dies."
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# -------------------------------------------------------------------------
# CORS Configuration (Restricted development & production origins)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# Structured Error Handlers (No raw stack traces exposed to client)
# -------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "error": {
                "code": "VALIDATION_ERROR",
                "message": "Request schema validation failed",
                "details": exc.errors()
            }
        }
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled server exception on %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": {
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred during inference or data processing",
                "details": str(exc)
            }
        }
    )

# -------------------------------------------------------------------------
# Mount Modular API Router
# -------------------------------------------------------------------------
app.include_router(api_router)

@app.get("/", summary="Root Health & Documentation Portal")
def root_portal():
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "online" if state.is_initialized else "initializing",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "api_prefix": "/api/v1",
        "canonical_split": "128 Train / 32 Val (27,972 dies)"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
