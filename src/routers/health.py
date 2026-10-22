from fastapi import APIRouter, status
from src.schemas.prediction import HealthResponse
from src.services.ml_service import check_model_health

router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse, status_code=status.HTTP_200_OK)
def health_check():
    """
    Fase 5 (Endpoint de Monitoramento):
    Verifica a integridade operacional da API e confirma se o modelo preditivo
    de Machine Learning reside ativamente na memória RAM.
    """
    model_ready = check_model_health()
    return HealthResponse(
        status="online",
        service="API de Manutenção Industrial Preditiva",
        model_loaded=model_ready,
        mode="production" if model_ready else "training_required"
    )
