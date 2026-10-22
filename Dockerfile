# Imagem base oficial do Python
FROM python:3.10-slim

# Evita criação de arquivos .pyc e força flush imediato de logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Instala dependências do sistema necessárias se houver
RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

# Copia dependências primeiro para aproveitar o cache do Docker
COPY requirements.txt .

# Instala as dependências Python
RUN pip install --no-cache-dir -r requirements.txt

# Copia o código da aplicação
COPY . .

# Treina o modelo inicial para empacotar o container pronto para produção
RUN python src/services/ml_service.py

# Expõe a porta que o FastAPI utiliza
EXPOSE 8000

# Execução do servidor ASGI
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
