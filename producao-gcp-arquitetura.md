# Arquitetura de Produção em Escala — Ecossistema Google Cloud (GCP)
## Manutenção Preditiva Industrial (Estimativa de RUL)

Este documento detalha o mapeamento e a evolução da solução local de **Manutenção Industrial Preditiva** (FastAPI, Scikit-Learn, SQLite) para uma arquitetura *cloud-native* gerenciada e altamente escalável no ecossistema **Google Cloud Platform (GCP)**.

---

## 1. Visão Geral da Transição de Arquitetura

| Componente da Aplicação | Arquitetura Local Atual | Arquitetura GCP em Escala |
| :--- | :--- | :--- |
| **Ingestão de Telemetria** | Requisições síncronas `POST /predict` | **Google Cloud Pub/Sub** (Ingestão de eventos IoT em tempo real) |
| **Processamento de Janela Móvel** | Pandas em memória | **Google Cloud Dataflow** (Apache Beam para janelas de 6h) |
| **Servimento da API REST** | FastAPI + Uvicorn (local) | **Google Cloud Run** (Containers autoescaláveis) ou **GKE** |
| **Banco de Dados Relacional** | SQLite (`prediction_logs.db`) | **Cloud SQL for PostgreSQL** (HA e conexões concorrentes) |
| **Controle de Idempotência & Cache** | Tabela SQLite | **Google Cloud Memorystore for Redis** |
| **Armazenamento de Histórico** | Arquivos CSV locais em `/data` | **Google Cloud Storage (GCS)** + **BigQuery** (Data Warehouse) |
| **Registro de Modelos & MLOps** | MLflow local + `models/model.joblib` | **Vertex AI Model Registry** + **Cloud Storage** |
| **Monitoramento & Alertas** | SMTP síncrono local | **Cloud Tasks** + **SendGrid/Cloud Functions** + **Cloud Monitoring** |
| **Segurança & Segredos** | Variáveis de ambiente locais | **Secret Manager** + **Identity-Aware Proxy (IAP)** |

---

## 2. Pilares de Arquitetura no GCP

### 2.1. Ingestão de Dados e Streaming IoT
- **Cloud Pub/Sub:** Atua como o *buffer* de ingestão global e desacoplado para todas as leituras de sensores (tensão, rotação, pressão e vibração) originadas dos motores na fábrica.
- **Cloud Dataflow (Apache Beam):** Processa o fluxo contínuo de telemetria, calculando as agregações de 6 horas, contagem de erros das últimas 24h e integração de históricos sem vazamento de dados (*data leakage*).

### 2.2. Hospedagem da API e Autoescalamento
- **Cloud Run:** Hospeda o container da API FastAPI. Oferece dimensionamento automático de zero até centenas de instâncias com base no volume de requisições HTTP, cobrando apenas pelo tempo de execução.
- **Carregamento do Modelo:** O artefato `model.joblib` (ou modelo compilado em formato ONNX/TensorFlow) é baixado diretamente do **Google Cloud Storage (GCS)** na inicialização do container (evento de *Lifespan* `@asynccontextmanager`), mantendo a inferência de altíssima velocidade em RAM.
- **Cloud Load Balancing & Cloud Armor:** Balanceamento de carga global com terminação TLS, proteção contra ataques DDoS e WAF na borda.

### 2.3. Persistência de Dados, Cache e Data Warehouse
- **Cloud SQL for PostgreSQL:** Armazena os registros de predição (`prediction_logs`), status dos motores e tabelas operacionais com replicação regional e backups automáticos.
- **Memorystore for Redis:** Gerencia o controle de idempotência (UUID v4 na `Idempotency-Key`) com tempo de resposta sub-militermilissegundo, além de guardar em cache predições recentes para evitar chamadas redundantes ao modelo.
- **BigQuery:** Data Warehouse serverless onde todos os logs e predições são espelhados em tempo real para análises estatísticas corporativas, BI (Looker) e retreinamento de modelos.

### 2.4. MLOps, Monitoramento e Detecção de Drift
- **Vertex AI Model Registry & Pipelines:** Centraliza o versionamento de artefatos de Machine Learning, metadados de experimentos e orquestração de retreinamentos automatizados.
- **Vertex AI Model Monitoring:** Monitora continuamente os dados de entrada em produção para identificar **Data Drift** (mudanças no comportamento dos sensores com o desgaste físico dos motores) e **Concept Drift**, disparando alertas de recalibração.
- **Cloud Build & Artifact Registry:** Pipeline de CI/CD automatizado via GitHub Actions que compila a imagem Docker a cada *merge* na branch `main` e implanta automaticamente no Cloud Run.

### 2.5. Resiliência de Alertas e Segurança
- **Cloud Tasks & Cloud Functions:** O disparo de e-mails para os estados **ALERTA** e **CRÍTICO** deixa de ser síncrono. A API envia uma tarefa para o Cloud Tasks, que aciona uma Cloud Function responsável pela entrega assíncrona da mensagem.
- **Secret Manager:** Armazena com segurança chaves de API, credenciais do PostgreSQL e tokens SMTP.
- **Cloud Logging & Cloud Monitoring:** Centraliza a observabilidade da aplicação com métricas de latência, taxa de erros HTTP (ex: contagem de erros 422) e saúde dos serviços.

---

## 3. Roteiro de Migração Recomendado (Roadmap)

1. **Fase 1 (Empacotamento & Container):** Dockerizar a API FastAPI atual e parametrizar as conexões de banco via variáveis de ambiente.
2. **Fase 2 (Infraestrutura Serverless Básica):** Subir o banco Cloud SQL (PostgreSQL) e publicar o container da API no Cloud Run lendo o `model.joblib` de um bucket do Cloud Storage.
3. **Fase 3 (Desacoplamento & Cache):** Integrar o Memorystore (Redis) para a checagem de idempotência e o Cloud Tasks para envio de e-mails assíncronos.
4. **Fase 4 (MLOps & Analytics):** Integrar o envio de logs ao BigQuery e configurar o Vertex AI Model Monitoring para rastreamento de drift.
