from typing import List, Optional
from fastapi import APIRouter, Depends, Header, HTTPException, Query, status
from sqlalchemy import func
from sqlalchemy.orm import Session
from src.database.connection import get_db
from src.database.models import PredictionLog, EmailAlertLog
from src.schemas.prediction import (
    PredictionRequest,
    PredictionResponse,
    PredictionLogSchema,
    EmailAlertLogSchema,
    MotorsByRulBandResponse,
)
from src.services.ml_service import predict_rul
from src.services.email_service import trigger_maintenance_email_alert

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
    em horas, registra auditoria no SQLite com suporte a `Idempotency-Key` e dispara alerta
    automático por e-mail quando o status é `CRÍTICO` ou `ALERTA`.
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
                existing_alert = (
                    db.query(EmailAlertLog)
                    .filter(EmailAlertLog.prediction_id == existing_log.id)
                    .first()
                )
                return PredictionResponse(
                    machine_id=existing_log.machine_id,
                    predicted_rul_hours=existing_log.predicted_rul_hours,
                    status=existing_log.status,
                    recommendation=existing_log.recommendation or "",
                    email_alert_triggered=existing_alert is not None,
                    email_delivery_mode=existing_alert.delivery_mode if existing_alert else None,
                    email_recipient=existing_alert.recipient if existing_alert else None
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
            model_type=request.model_type,
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

        # 4. Automação de Alerta por E-mail quando o status for CRÍTICO ou ALERTA
        alert_entry = trigger_maintenance_email_alert(
            db=db,
            prediction_id=log_entry.id,
            machine_id=request.machine_id,
            model_type=request.model_type,
            predicted_rul_hours=prediction_result["predicted_rul_hours"],
            status=prediction_result["status"],
            recommendation=prediction_result["recommendation"],
            volt=request.volt,
            rotate=request.rotate,
            pressure=request.pressure,
            vibration=request.vibration,
            recent_errors=request.recent_errors,
            days_since_maint=request.days_since_maint,
            recipient_override=request.alert_email
        )

        return PredictionResponse(
            machine_id=request.machine_id,
            predicted_rul_hours=prediction_result["predicted_rul_hours"],
            status=prediction_result["status"],
            recommendation=prediction_result["recommendation"],
            email_alert_triggered=alert_entry is not None,
            email_delivery_mode=alert_entry.delivery_mode if alert_entry else None,
            email_recipient=alert_entry.recipient if alert_entry else None
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
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filtrar histórico por faixa de RUL: CRÍTICO, ALERTA ou NORMAL"
    ),
    db: Session = Depends(get_db)
):
    """
    Lista o histórico de predições de RUL registradas no banco SQLite.
    """
    query = db.query(PredictionLog)
    if machine_id is not None:
        query = query.filter(PredictionLog.machine_id == machine_id)
    if status_filter:
        normalized = status_filter.strip().upper()
        if normalized == "CRITICO":
            normalized = "CRÍTICO"
        query = query.filter(PredictionLog.status == normalized)
    return query.order_by(PredictionLog.id.desc()).limit(limit).all()


@router.get("/motors/status", response_model=MotorsByRulBandResponse, status_code=status.HTTP_200_OK)
def list_motors_by_rul_band(
    band: Optional[str] = Query(
        None,
        description="Faixa de RUL para filtro: CRÍTICO (<=48h), ALERTA (48h-168h) ou NORMAL (>168h)"
    ),
    db: Session = Depends(get_db)
):
    """
    Retorna o estado mais recente de cada motor único monitorado, agrupado e filtrado
    pelas 3 faixas de RUL:
    - `CRÍTICO`: RUL <= 48h
    - `ALERTA`: 48h < RUL <= 168h
    - `NORMAL`: RUL > 168h
    """
    subquery = (
        db.query(
            PredictionLog.machine_id,
            func.max(PredictionLog.id).label("max_id")
        )
        .group_by(PredictionLog.machine_id)
        .subquery()
    )

    latest_logs: List[PredictionLog] = (
        db.query(PredictionLog)
        .join(subquery, PredictionLog.id == subquery.c.max_id)
        .order_by(PredictionLog.predicted_rul_hours.asc())
        .all()
    )

    counts = {"CRÍTICO": 0, "ALERTA": 0, "NORMAL": 0}
    for log in latest_logs:
        if log.status in counts:
            counts[log.status] += 1

    if band:
        normalized_band = band.strip().upper()
        if normalized_band == "CRITICO":
            normalized_band = "CRÍTICO"
        filtered_motors = [log for log in latest_logs if log.status == normalized_band]
    else:
        filtered_motors = latest_logs

    return MotorsByRulBandResponse(
        total_motors=len(latest_logs),
        counts_by_band=counts,
        motors=filtered_motors
    )


@router.get("/alerts", response_model=List[EmailAlertLogSchema], status_code=status.HTTP_200_OK)
def list_email_alerts(
    limit: int = Query(20, ge=1, le=200, description="Quantidade máxima de alertas de e-mail retornados"),
    status_filter: Optional[str] = Query(
        None,
        alias="status",
        description="Filtrar alertas por criticidade (CRÍTICO ou ALERTA)"
    ),
    db: Session = Depends(get_db)
):
    """
    Lista todos os alertas de e-mail disparados automaticamente para motores em estado ALERTA ou CRÍTICO.
    """
    query = db.query(EmailAlertLog)
    if status_filter:
        normalized = status_filter.strip().upper()
        if normalized == "CRITICO":
            normalized = "CRÍTICO"
        query = query.filter(EmailAlertLog.status == normalized)
    return query.order_by(EmailAlertLog.id.desc()).limit(limit).all()
