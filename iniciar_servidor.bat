@echo off
title Iniciando Servidor Uvicorn e Pagina Web

echo Acessando o diretorio do projeto...
:: Muda automaticamente para o diretorio onde este script .bat esta localizado
cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo Iniciando o servidor Uvicorn...
:: O comando "start" abre o servidor em uma nova janela para nao travar o script
start "API Manutencao Preditiva" python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

echo Aguardando o servidor inicializar...
timeout /t 5 /nobreak >nul

echo Abrindo a pagina web no navegador padrao...
start http://127.0.0.1:8000/simulador

echo Concluido!
exit
