@echo off
chcp 65001 >nul
setlocal
cd /d "%~dp0"

echo ============================================================================
echo   MANUTENCAO INDUSTRIAL PREDITIVA - INICIALIZACAO AUTOMATICA (WINDOWS)
echo ============================================================================
echo.

:: 1. Verificar/Criar ambiente virtual (venv)
if not exist "venv\Scripts\activate.bat" (
    echo [1/4] Criando ambiente virtual Python ^(venv^)...
    python -m venv venv
    if errorlevel 1 (
        echo [ERRO] Falha ao criar o ambiente virtual. Verifique se o Python esta instalado e no PATH.
        pause
        exit /b 1
    )
) else (
    echo [1/4] Ambiente virtual ^(venv^) ja encontrado.
)

:: 2. Ativar ambiente virtual e instalar dependencias
echo [2/4] Ativando ambiente virtual e verificando dependencias...
call venv\Scripts\activate.bat
pip install -r requirements.txt --quiet
if errorlevel 1 (
    echo [ERRO] Falha ao instalar as dependencias do requirements.txt.
    pause
    exit /b 1
)

:: 3. Verificar artefato do modelo preditivo
if not exist "models\model.joblib" (
    echo [3/4] Artefato models\model.joblib nao encontrado. Executando treinamento...
    python src/services/ml_service.py
) else (
    echo [3/4] Artefato preditivo ^(models\model.joblib^) verificado com sucesso.
)

:: 4. Executar suite de testes automatizados
echo [4/4] Executando suite de testes automatizados ^(Pytest^)...
pytest -v tests/
if errorlevel 1 (
    echo [AVISO] Algum teste apresentou falha. Verifique o log acima.
)

echo.
echo ============================================================================
echo   SERVIDOR FASTAPI PRONTO PARA INICIAR!
echo   - Swagger UI (/docs):      http://127.0.0.1:8000/docs
echo   - Painel Simulador Web:    http://127.0.0.1:8000/simulador
echo   - Health Check (/health):  http://127.0.0.1:8000/health
echo ============================================================================
echo.

python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
pause
