@echo off
call venv\Scripts\activate

title Iniciando Servidor Uvicorn e Pagina Web

echo Acessando o diretorio do projeto...
:: O parametro /d garante a mudanca de disco (de C: para D:) e de pasta ao mesmo tempo
cd /d "D:\ANTIGRAVITY\PROJETOS\manutencao_industrial_preditiva"

echo Iniciando o servidor Uvicorn...
:: O comando "start" abre o servidor em uma nova janela para nao travar o script
start uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload

echo Aguardando o servidor inicializar...
timeout /t 10 /nobreak >nul

echo Abrindo a pagina web no navegador padrao...
start http://127.0.0.1:8000/simulador

echo Concluido!
exit

