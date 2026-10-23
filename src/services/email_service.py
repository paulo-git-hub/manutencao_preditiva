import os
import smtplib
from email.message import EmailMessage
from pathlib import Path
from typing import Dict, Optional
from sqlalchemy.orm import Session
from src.database.models import EmailAlertLog

BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = BASE_DIR / ".env"


def get_smtp_config() -> Dict[str, str]:
    """
    Lê as configurações de SMTP diretamente do arquivo `.env` na raiz do projeto
    em tempo real (sem exigir reinicialização do servidor) ou das variáveis de ambiente.
    """
    file_vars: Dict[str, str] = {}
    if ENV_PATH.exists():
        for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, val = line.split("=", 1)
            file_vars[key.strip()] = val.strip().strip('"').strip("'")

    def _get(key: str, default: str = "") -> str:
        return file_vars.get(key) or os.getenv(key, default).strip()

    smtp_host = _get("SMTP_HOST", "")
    # Corrige automaticamente caso o usuário tenha digitado smtp.google.com
    if smtp_host.lower() == "smtp.google.com":
        smtp_host = "smtp.gmail.com"

    return {
        "SMTP_HOST": smtp_host,
        "SMTP_PORT": _get("SMTP_PORT", "587"),
        "SMTP_USER": _get("SMTP_USER", ""),
        "SMTP_PASSWORD": _get("SMTP_PASSWORD", ""),
        "ALERT_EMAIL_TO": _get("ALERT_EMAIL_TO", "manutencao.pcm@industria-senai.com.br"),
    }


def trigger_maintenance_email_alert(
    db: Session,
    prediction_id: Optional[int],
    machine_id: int,
    model_type: str,
    predicted_rul_hours: float,
    status: str,
    recommendation: str,
    volt: float,
    rotate: float,
    pressure: float,
    vibration: float,
    recent_errors: int,
    days_since_maint: float,
    recipient_override: Optional[str] = None
) -> Optional[EmailAlertLog]:
    """
    Dispara alerta automatizado de manutenção por e-mail sempre que o motor atingir
    status 'CRÍTICO' (RUL <= 48h) ou 'ALERTA' (48h < RUL <= 168h).
    """
    if status not in ("CRÍTICO", "ALERTA"):
        return None

    cfg = get_smtp_config()
    smtp_host = cfg["SMTP_HOST"]
    smtp_port = int(cfg["SMTP_PORT"] or "587")
    smtp_user = cfg["SMTP_USER"]
    smtp_password = cfg["SMTP_PASSWORD"]
    default_recipient = cfg["ALERT_EMAIL_TO"]
    recipient = (recipient_override or default_recipient).strip()

    urgency_prefix = "🚨 [URGENTE - CRÍTICO]" if status == "CRÍTICO" else "⚠️ [ATENÇÃO - ALERTA]"
    subject = (
        f"{urgency_prefix} Motor #{machine_id} ({model_type}) — "
        f"RUL Estimada: {predicted_rul_hours:.1f}h ({status})"
    )

    body = (
        f"RELATÓRIO AUTOMÁTICO DE MANUTENÇÃO PREDITIVA (PCM)\n"
        f"==================================================\n\n"
        f"• Identificador do Ativo : Motor #{machine_id}\n"
        f"• Família Construtiva    : {model_type}\n"
        f"• Classificação de Risco : {status}\n"
        f"• Vida Útil Restante     : {predicted_rul_hours:.2f} horas\n"
        f"• Recomendação Técnica   : {recommendation}\n\n"
        f"TELEMETRIA COLETADA NO MOMENTO DA INFERÊNCIA:\n"
        f"--------------------------------------------------\n"
        f"• Tensão Elétrica  : {volt:.1f} V\n"
        f"• Rotação (RPM)    : {rotate:.1f} RPM\n"
        f"• Pressão          : {pressure:.1f} psi\n"
        f"• Vibração         : {vibration:.1f} m/s²\n"
        f"• Erros (24h)      : {recent_errors}\n"
        f"• Dias s/ Manut.   : {days_since_maint:.1f} dias\n\n"
        f"Este alerta foi gerado automaticamente pelo Pipeline de IA de Manutenção Preditiva."
    )

    delivery_mode = "SIMULATED_INBOX"

    if smtp_host and smtp_user and smtp_password:
        try:
            msg = EmailMessage()
            msg["Subject"] = subject
            msg["From"] = smtp_user
            msg["To"] = recipient
            msg.set_content(body)

            with smtplib.SMTP(smtp_host, smtp_port, timeout=15) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(smtp_user, smtp_password)
                server.send_message(msg)

            delivery_mode = "SMTP_SENT"
        except Exception as exc:
            print(f"[AVISO SMTP] Falha ao enviar e-mail real ({exc}). Salvando como SMTP_FALLBACK.")
            delivery_mode = "SMTP_FALLBACK"

    alert_entry = EmailAlertLog(
        prediction_id=prediction_id,
        machine_id=machine_id,
        model_type=model_type,
        predicted_rul_hours=predicted_rul_hours,
        status=status,
        recipient=recipient,
        subject=subject,
        body=body,
        delivery_mode=delivery_mode
    )
    db.add(alert_entry)
    db.commit()
    db.refresh(alert_entry)
    return alert_entry
