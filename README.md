# 🛠️ Manutenção Industrial Preditiva — Estimativa de RUL (SENAI / SCTEC)

> **Situação de Aprendizagem (Projeto Avaliativo) — Módulo 2 - Semana 13**  
> **Desenvolvimento de IA para Análise Preditiva [T1 e T2]**

---

## 1. Nome e Resumo do Projeto

**Manutenção Industrial Preditiva com Estimativa de Vida Útil Restante (RUL - *Remaining Useful Life*)**

Na indústria de manufatura contínua, paradas não planejadas de motores e ativos rotativos geram severos prejuízos operacionais, enquanto manutenções preventivas baseadas apenas em calendário trocam componentes ainda saudáveis.

Este projeto resolve esse problema de negócio disponibilizando um **pipeline preditivo de Machine Learning em produção via API RESTful (FastAPI)**: a partir de leituras de sensores em tempo real (tensão elétrica, rotação, pressão e vibração), somadas à idade do ativo, família construtiva do motor, histórico de erros nas últimas 24h e dias desde a última manutenção, o sistema estima quantas **horas de vida útil restam até a próxima falha (RUL)**, classifica o risco operacional (**NORMAL**, **ALERTA** ou **CRÍTICO**), recomenda a ação de engenharia adequada, dispara **alertas automáticos por e-mail** para os estados **ALERTA** e **CRÍTICO**, e registra os logs de auditoria em banco **SQLite** com garantia de **Idempotência (`Idempotency-Key`)**.

---

## 2. Origem do Modelo e Relatório de Métricas (Fase 1)

### Opção Escolhida: **Opção B (Pipeline Preditivo Próprio a partir de Base Pública Justificada)**
* **Base de Dados Pública:** **Microsoft Azure Predictive Maintenance Dataset** (disponível publicamente no Kaggle / Azure AI Gallery).
* **Composição dos Dados (`/data`):** 5 tabelas relacionais integradas temporalmente sem vazamento de dados (*data leakage*) via `pd.merge_asof`:
  1. `PdM_telemetry.csv`: Séries temporais horárias de tensão (`volt`), rotação (`rotate`), pressão (`pressure`) e vibração (`vibration`).
  2. `PdM_machines.csv`: Cadastro da idade (`age`) e família construtiva (`model1`, `model2`, `model3`, `model4`).
  3. `PdM_failures.csv`: Histórico de quebras reais para cálculo regressivo do alvo `rul_hours`.
  4. `PdM_errors.csv`: Alertas operacionais não-fatais agrupados em janela móvel de 24h (`recent_errors`).
  5. `PdM_maint.csv`: Intervenções anteriores para cálculo dos dias desde a última manutenção (`days_since_maint`).

### Tratamento, Divisão de Dados e Pipeline Serializado
* **Divisão Estrita:** `42.542` janelas temporais consolidadas, divididas em **Treino (70%)**, **Validação (15%)** e **Teste Holdout (15%)**.
* **Pré-processamento Automatizado (`ColumnTransformer` + `Pipeline`):**
  * **Variáveis Numéricas (`volt`, `rotate`, `pressure`, `vibration`, `age`, `recent_errors`, `days_since_maint`):** Imputação de nulos pela mediana (`SimpleImputer(strategy="median")`) + padronização z-score (`StandardScaler()`).
  * **Variáveis Categóricas (`model_type`):** Imputação pela moda (`SimpleImputer(strategy="most_frequent")`) + codificação binária (`OneHotEncoder(handle_unknown="ignore")`).
  * **Estimador:** `RandomForestRegressor(n_estimators=120, max_depth=14, random_state=42)`.
* **Serialização e Rastreamento:** Pipeline completo exportado em [`models/model.joblib`](models/model.joblib), métricas salvas em [`models/metrics_report.json`](models/metrics_report.json) e experimento versionado via **MLflow**.

### 📈 Relatório de Métricas Alcançadas

| Conjunto de Avaliação | MSE ($\text{horas}^2$) | RMSE ($\text{horas}$) | MAE ($\text{horas}$) | Coeficiente de Determinação ($R^2$) |
| :--- | :---: | :---: | :---: | :---: |
| **Validação (15%)** | `41.250,86` | `203,10 h` | `168,97 h` | **`0,3199`** |
| **Teste / Holdout (15%)** | `40.280,18` | `200,70 h` | `166,65 h` | **`0,3407`** |

---

## 3. Tecnologias Utilizadas

| Biblioteca / Ferramenta | Versão Especificada | Papel na Arquitetura |
| :--- | :---: | :--- |
| **Python** | `3.10+` | Linguagem base do pipeline de Ciência de Dados e API Web |
| **FastAPI** | `>=0.100.0` | Framework web assíncrono para exposição dos endpoints REST e Lifespan |
| **Uvicorn** | `>=0.22.0` | Servidor ASGI de alta performance para execução em produção |
| **Pydantic** | `>=2.0.0` | Modelagem e validação estrita de esquemas de entrada/saída (`BaseModel`, `Field`) |
| **Scikit-Learn** | `>=1.3.0` | Construção do `Pipeline`, `ColumnTransformer` e `RandomForestRegressor` |
| **Pandas & NumPy** | `>=2.0.0` / `>=1.24.0` | Engenharia de atributos vetorizada (`merge_asof`) e manipulação matricial |
| **Joblib** | `>=1.3.0` | Serialização e carregamento em memória RAM do artefato `models/model.joblib` |
| **SQLAlchemy (SQLite)** | `>=2.0.0` | Persistência relacional de logs de inferência, alertas e controle de idempotência |
| **MLflow** | `>=2.5.0` | Rastreamento de hiperparâmetros, métricas e registro do modelo |
| **Matplotlib & Seaborn** | `>=3.7.0` / `>=0.12.0` | Visualização gráfica e matriz de correlação no Notebook de EDA |
| **Pytest & HTTPX** | `>=7.4.0` / `>=0.24.0` | Suíte de testes automatizados de integração dos endpoints e validações HTTP |

---

## 4. Instruções de Instalação e Execução

### 4.1 Criação do Ambiente Virtual (`venv`) e Instalação das Dependências

```bash
# 1. Clonar o repositório e acessar a pasta raiz
git clone <URL_DO_SEU_REPOSITORIO_GITHUB>
cd manutencao_industrial_preditiva

# 2. Criar o ambiente virtual Python
python -m venv venv

# 3. Ativar o ambiente virtual
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Linux / macOS:
source venv/bin/activate

# 4. Instalar todas as dependências listadas
pip install -r requirements.txt
```

### 4.2 (Opcional) Reexecutar o Pipeline Offline de Treinamento e MLflow

O repositório já inclui o artefato treinado em `models/model.joblib`. Caso deseje retreinar o pipeline do zero:

```bash
python src/services/ml_service.py
```

### 4.3 Inicialização do Servidor ASGI com Uvicorn

```bash
python -m uvicorn src.main:app --host 127.0.0.1 --port 8000 --reload
```

### 4.4 Executar a Suíte de Testes Automatizados (Pytest)

```bash
pytest -v tests/
```

---

## 5. Documentação da API e Arquitetura Modular (Fases 2 a 5)

### Estrutura em Camadas do Projeto

```text
/manutencao_industrial_preditiva
│-- /data                      # 5 datasets CSV da Azure Predictive Maintenance (caminhos relativos)
│-- /models                    # Artefato binário serializado (model.joblib) e relatório (metrics_report.json)
│-- /notebooks                 # Notebook estruturado por Fases (01_eda_and_training.ipynb)
│-- /src                       # Código-fonte modularizado da API FastAPI
│   │-- /database              # Conexão SQLite e ORM SQLAlchemy (connection.py, models.py)
│   │-- /schemas               # Contratos Pydantic de entrada e saída (prediction.py)
│   │-- /services              # Pipeline ML, cache em RAM e alertas SMTP (ml_service.py, email_service.py)
│   │-- /routers               # Rotas modulares APIRouter (health.py, predict.py)
│   └── main.py                # Ponto de entrada e gerenciamento de ciclo de vida (@asynccontextmanager lifespan)
│-- /tests                     # Testes automatizados (test_api.py)
│-- apresentacao.html          # Slides interativos guiados para o vídeo (Item 5.4)
│-- Dockerfile                 # Empacotamento em container
│-- requirements.txt           # Dependências com versões
└── README.md                  # Documentação técnica
```

### Links Interativos da Aplicação (Servidor Local)
* **Documentação Interativa Swagger UI (`/docs`):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
* **Documentação Alternativa ReDoc (`/redoc`):** [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)
* **Painel Simulador Web com Gráfico de Degradação (`/simulador`):** [http://127.0.0.1:8000/simulador](http://127.0.0.1:8000/simulador)
* **Monitoramento de Saúde e Modelo em RAM (`/health`):** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)
* **Monitoramento de Motores pelas 3 Faixas de RUL (`/motors/status`):** [http://127.0.0.1:8000/motors/status](http://127.0.0.1:8000/motors/status)
* **Histórico de Alertas Automáticos de E-mail (`/alerts`):** [http://127.0.0.1:8000/alerts](http://127.0.0.1:8000/alerts)

### Exemplos de Chamadas aos Endpoints

#### 1. Checagem de Saúde (`GET /health`) — Resposta `200 OK`:
```json
{
  "status": "online",
  "service": "API de Manutenção Industrial Preditiva",
  "model_loaded": true,
  "mode": "production"
}
```

#### 2. Inferência Preditiva (`POST /predict` ou `POST /v1/predict`) — Payload JSON de Entrada:
```json
{
  "machine_id": 1,
  "model_type": "model1",
  "volt": 172.5,
  "rotate": 418.0,
  "pressure": 115.2,
  "vibration": 46.8,
  "age": 14.0,
  "recent_errors": 2,
  "days_since_maint": 25.0,
  "alert_email": "manutencao.pcm@industria-senai.com.br"
}
```

#### Resposta (`200 OK` — `PredictionResponse`):
```json
{
  "machine_id": 1,
  "predicted_rul_hours": 142.85,
  "status": "ALERTA",
  "recommendation": "Agendar inspeção e lubrificação/troca de peças no próximo ciclo de manutenção semanal.",
  "email_alert_triggered": true,
  "email_delivery_mode": "SIMULATED_INBOX",
  "email_recipient": "manutencao.pcm@industria-senai.com.br"
}
```

#### 3. Exemplo de Interceptação de Dados Inválidos (`422 Unprocessable Entity`):
Envio de tensão negativa (`"volt": -170.0`) ou texto inválido é bloqueado automaticamente pelo Pydantic antes de atingir o modelo preditivo.

---

## 6. Links das Evidências (Entrega AVA)

* 🎥 **Link do Vídeo de Apresentação (Google Drive — Modo Leitor):** `[INSIRA_AQUI_O_LINK_DO_SEU_VIDEO_NO_GOOGLE_DRIVE]`
* 💻 **Link do Repositório Público no GitHub:** `[INSIRA_AQUI_O_LINK_DO_SEU_REPOSITORIO_NO_GITHUB]`
