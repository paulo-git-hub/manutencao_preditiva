from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, declarative_base

DATABASE_URL = "sqlite:///./predictions.db"

engine = create_engine(
    DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def ensure_schema_columns():
    """Garante migração leve no SQLite caso a tabela prediction_logs já exista sem a coluna model_type."""
    with engine.connect() as conn:
        res = conn.execute(text("PRAGMA table_info(prediction_logs)")).fetchall()
        existing_cols = {row[1] for row in res}
        if res and "model_type" not in existing_cols:
            conn.execute(text("ALTER TABLE prediction_logs ADD COLUMN model_type VARCHAR(16) DEFAULT 'model1'"))
            conn.commit()


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
