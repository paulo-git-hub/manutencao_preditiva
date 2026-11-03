#!/usr/bin/env bash
set -e

# Garantir execucao a partir da pasta raiz do projeto
cd "$(dirname "$0")"

echo "============================================================================"
echo "  MANUTENCAO INDUSTRIAL PREDITIVA - INICIALIZACAO AUTOMATICA (LINUX / BASH)"
echo "============================================================================"
echo ""

# Detectar interpretador Python (python3 no Linux/macOS ou python no Git Bash)
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "[ERRO] Python nao encontrado no sistema. Instale o Python 3.10+ antes de continuar."
    exit 1
fi

# 1. Verificar/Criar ambiente virtual (venv)
if [ ! -f "venv/bin/activate" ] && [ ! -f "venv/Scripts/activate" ]; then
    echo "[1/4] Criando ambiente virtual Python (venv) com $PYTHON_CMD..."
    $PYTHON_CMD -m venv venv
else
    echo "[1/4] Ambiente virtual (venv) ja encontrado."
fi

# 2. Ativar ambiente virtual (compativel com Linux/macOS e Git Bash MINGW64 no Windows)
echo "[2/4] Ativando ambiente virtual e verificando dependencias..."
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
elif [ -f "venv/Scripts/activate" ]; then
    source venv/Scripts/activate
else
    echo "[ERRO] Script de ativacao do venv nao encontrado."
    exit 1
fi

pip install -r requirements.txt --quiet

# 3. Verificar artefato do modelo preditivo
if [ ! -f "models/model.joblib" ]; then
    echo "[3/4] Artefato models/model.joblib nao encontrado. Executando treinamento..."
    python src/services/ml_service.py
else
    echo "[3/4] Artefato preditivo (models/model.joblib) verificado com sucesso."
fi

# 4. Executar suite de testes automatizados
echo "[4/4] Executando suite de testes automatizados (Pytest)..."
pytest -v tests/

echo ""
echo "============================================================================"
echo "  SERVIDOR FASTAPI PRONTO PARA INICIAR!"
echo "  - Swagger UI (/docs):      http://127.0.0.1:8000/docs"
echo "  - Painel Simulador Web:    http://127.0.0.1:8000/simulador"
echo "  - Health Check (/health):  http://127.0.0.1:8000/health"
echo "============================================================================"
echo ""

python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
