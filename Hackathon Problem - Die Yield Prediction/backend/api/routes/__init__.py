from fastapi import APIRouter
from backend.api.routes.system import router as system_router
from backend.api.routes.inference import router as inference_router
from backend.api.routes.wafers import router as wafers_router
from backend.api.routes.models import router as models_router
from backend.api.routes.evaluation import router as evaluation_router

api_router = APIRouter(prefix="/api/v1")

api_router.include_router(system_router)
api_router.include_router(inference_router)
api_router.include_router(wafers_router)
api_router.include_router(models_router)
api_router.include_router(evaluation_router)

__all__ = ["api_router"]
