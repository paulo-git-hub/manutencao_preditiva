from datetime import datetime, timezone
from sqlalchemy import Column, Integer, Float, String, DateTime, Text
from .connection import Base


def utc_now():
    return datetime.now(timezone.utc)


class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)
    idempotency_key = Column(String(128), unique=True, index=True, nullable=True)
    machine_id = Column(Integer, index=True)
    model_type = Column(String(16), default="model1")
    volt = Column(Float)
    rotate = Column(Float)
    pressure = Column(Float)
    vibration = Column(Float)
    age = Column(Float)
    recent_errors = Column(Integer, default=0)
    days_since_maint = Column(Float, default=15.0)
    predicted_rul_hours = Column(Float)
    status = Column(String(50), index=True)
    recommendation = Column(String(255))
    timestamp = Column(DateTime, default=utc_now)


class EmailAlertLog(Base):
    __tablename__ = "email_alert_logs"

    id = Column(Integer, primary_key=True, index=True)
    prediction_id = Column(Integer, index=True, nullable=True)
    machine_id = Column(Integer, index=True)
    model_type = Column(String(16), default="model1")
    predicted_rul_hours = Column(Float)
    status = Column(String(50), index=True)
    recipient = Column(String(255))
    subject = Column(String(255))
    body = Column(Text)
    delivery_mode = Column(String(50))  # "SMTP_SENT", "SIMULATED_INBOX", "SMTP_FALLBACK"
    timestamp = Column(DateTime, default=utc_now)
