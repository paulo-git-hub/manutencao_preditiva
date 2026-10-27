from datetime import datetime
from typing import Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


class HealthResponse(BaseModel):
    status: str = Field(..., description="Estado operacional do serviço ASGI (online/degraded)")
    service: str = Field(..., description="Nome identificador do microsserviço")
    model_loaded: bool = Field(..., description="Indica se o artefato preditivo reside ativamente na memória RAM")
    mode: str = Field(..., description="Modo de operação atual da API")


class PredictionRequest(BaseModel):
    machine_id: int = Field(
        ...,
        gt=0,
        le=10000,
        description="Identificador único da máquina/motor industrial (1 a 10000)",
        json_schema_extra={"example": 1}
    )
    model_type: str = Field(
        default="model1",
        min_length=6,
        max_length=6,
        pattern=r"^model[1-4]$",
        description="Família construtiva do motor segundo catálogo Azure (model1, model2, model3 ou model4)",
        json_schema_extra={"example": "model1"}
    )
    volt: float = Field(
        ...,
        gt=0,
        le=500.0,
        description="Tensão elétrica média em operação (Volts, 0 < volt <= 500)",
        json_schema_extra={"example": 170.0}
    )
    rotate: float = Field(
        ...,
        gt=0,
        le=2000.0,
        description="Velocidade de rotação média do rotor (RPM, 0 < rotate <= 2000)",
        json_schema_extra={"example": 420.0}
    )
    pressure: float = Field(
        ...,
        gt=0,
        le=500.0,
        description="Pressão média de operação (psi, 0 < pressure <= 500)",
        json_schema_extra={"example": 105.0}
    )
    vibration: float = Field(
        ...,
        ge=0,
        le=300.0,
        description="Nível de vibração médio no mancal (m/s², 0 <= vibration <= 300)",
        json_schema_extra={"example": 42.0}
    )
    age: float = Field(
        default=10.0,
        ge=0,
        le=100.0,
        description="Idade do equipamento em anos (0 a 100)",
        json_schema_extra={"example": 14.0}
    )
    recent_errors: int = Field(
        default=0,
        ge=0,
        le=500,
        description="Quantidade de erros registrados nas últimas 24h (0 a 500)",
        json_schema_extra={"example": 1}
    )
    days_since_maint: float = Field(
        default=15.0,
        ge=0,
        le=3650.0,
        description="Dias decorridos desde a última manutenção (0 a 3650)",
        json_schema_extra={"example": 12.5}
    )
    alert_email: Optional[str] = Field(
        default=None,
        description="E-mail opcional de destino para receber alerta automático caso o status seja ALERTA ou CRÍTICO",
        json_schema_extra={"example": "paulo@sbsti.com"}
    )


class PredictionResponse(BaseModel):
    record_id: Optional[int] = Field(
        default=None,
        description="ID do registro persistido na tabela prediction_logs do SQLite"
    )
    idempotency_key: Optional[str] = Field(
        default=None,
        description="Chave de idempotência (UUID v4 gerado automaticamente ou chave enviada pelo cliente)"
    )
    machine_id: int = Field(..., gt=0, description="Identificador da máquina avaliada")
    predicted_rul_hours: float = Field(..., ge=0, description="Vida útil restante estimada em horas (RUL)")
    status: str = Field(..., description="Classificação de criticidade: CRÍTICO, ALERTA ou NORMAL")
    recommendation: str = Field(..., description="Ação preventiva recomendada para a equipe de manutenção")
    email_alert_triggered: bool = Field(
        default=False,
        description="Indica se um alerta automático via e-mail foi disparado (para status CRÍTICO ou ALERTA)"
    )
    email_delivery_mode: Optional[str] = Field(
        default=None,
        description="Modo de entrega do alerta de e-mail (SMTP_SENT, SIMULATED_INBOX ou None)"
    )
    email_recipient: Optional[str] = Field(
        default=None,
        description="E-mail destinatário efetivamente utilizado no alerta"
    )


class PredictionLogSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    idempotency_key: Optional[str] = None
    machine_id: int
    model_type: Optional[str] = "model1"
    volt: float
    rotate: float
    pressure: float
    vibration: float
    age: float
    recent_errors: int
    days_since_maint: float
    predicted_rul_hours: float
    status: str
    recommendation: Optional[str] = None
    timestamp: datetime


class EmailAlertLogSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    prediction_id: Optional[int] = None
    machine_id: int
    model_type: Optional[str] = "model1"
    predicted_rul_hours: float
    status: str
    recipient: str
    subject: str
    body: str
    delivery_mode: str
    timestamp: datetime


class MotorsByRulBandResponse(BaseModel):
    total_motors: int = Field(..., description="Quantidade total de motores únicos monitorados")
    counts_by_band: Dict[str, int] = Field(
        ...,
        description="Contagem de motores em cada uma das 3 faixas de RUL (CRÍTICO, ALERTA, NORMAL)"
    )
    motors: List[PredictionLogSchema] = Field(
        ...,
        description="Lista do estado mais recente de cada motor filtrado pela faixa de RUL"
    )
