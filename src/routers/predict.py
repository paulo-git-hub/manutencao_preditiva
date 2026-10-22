from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import PredictionLog
from src.schemas.prediction import PredictionRequest, PredictionResponse, PredictionLogSchema
from src.services.ml_service import predict_rul

router = APIRouter(tags=["Prediction"])


@router.post("/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK)
@router.post("/v1/predict", response_model=PredictionResponse, status_code=status.HTTP_200_OK, include_in_schema=True)
def predict(
    request: PredictionRequest,
    idempotency_key: Optional[str] = Header(
        default=None,
        alias="Idempotency-Key",
        description="Chave única opcional para garantir idempotência em retentativas de requisição"
    ),
    db: Session = Depends(get_db)
):
    """
    Fase 5 (Endpoint de Inferência):
    Recebe medições de sensores validadas via Pydantic, invoca a camada de serviço (`ml_service`)
    utilizando o modelo carregado em memória RAM (`lifespan`), estima a Vida Útil Restante (RUL)
    em horas e registra auditoria no SQLite com suporte a `Idempotency-Key`.
    """
    try:
        # 1. Verificação de Idempotência: se a chave já foi processada, retorna o registro existente
        if idempotency_key:
            existing_log = (
                db.query(PredictionLog)
                .filter(PredictionLog.idempotency_key == idempotency_key)
                .first()
            )
            if existing_log:
                return PredictionResponse(
                    machine_id=existing_log.machine_id,
                    predicted_rul_hours=existing_log.predicted_rul_hours,
                    status=existing_log.status,
                    recommendation=existing_log.recommendation or ""
                )

        # 2. Executa a inferência matemática na camada de serviço (modelo em RAM)
        prediction_result = predict_rul(
            volt=request.volt,
            rotate=request.rotate,
            pressure=request.pressure,
            vibration=request.vibration,
            age=request.age,
            recent_errors=request.recent_errors,
            days_since_maint=request.days_since_maint,
            model_type=request.model_type
        )

        # 3. Registro de auditoria no banco de dados SQLite
        log_entry = PredictionLog(
            idempotency_key=idempotency_key,
            machine_id=request.machine_id,
            volt=request.volt,
            rotate=request.rotate,
            pressure=request.pressure,
            vibration=request.vibration,
            age=request.age,
            recent_errors=request.recent_errors,
            days_since_maint=request.days_since_maint,
            predicted_rul_hours=prediction_result["predicted_rul_hours"],
            status=prediction_result["status"],
            recommendation=prediction_result["recommendation"]
        )
        db.add(log_entry)
        db.commit()
        db.refresh(log_entry)

        return PredictionResponse(
            machine_id=request.machine_id,
            predicted_rul_hours=prediction_result["predicted_rul_hours"],
            status=prediction_result["status"],
            recommendation=prediction_result["recommendation"]
        )

    except ValueError as ve:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Requisição inválida nos parâmetros de negócio: {str(ve)}"
        )
    except (FileNotFoundError, RuntimeError) as fnf:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Modelo preditivo indisponível em memória RAM: {str(fnf)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erro interno no processamento da predição: {str(e)}"
        )


@router.get("/predictions", response_model=List[PredictionLogSchema], status_code=status.HTTP_200_OK)
def list_predictions(
    limit: int = Query(50, ge=1, le=500, description="Quantidade máxima de registros retornados"),
    machine_id: Optional[int] = Query(None, gt=0, description="Filtrar por ID da máquina"),
    db: Session = Depends(get_db)
):
    """
    Lista o histórico de predições de RUL registradas no banco SQLite.
    """
    query = db.query(PredictionLog)
    if machine_id is not None:
        query = query.filter(PredictionLog.machine_id == machine_id)
    return query.order_by(PredictionLog.id.desc()).limit(limit).all()
