from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.routers import health, predict
from src.database.connection import engine, Base, ensure_schema_columns
from src.services.ml_service import load_model_on_startup, clear_model_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fase 3 (Ciclo de Vida e Inicialização Otimizada - Lifespan):
    Inicializa as tabelas do banco SQLite e carrega o artefato serializado (models/model.joblib)
    uma única vez na memória RAM durante o startup do servidor ASGI.
    """
    Base.metadata.create_all(bind=engine)
    ensure_schema_columns()
    model = load_model_on_startup()
    app.state.model = model
    yield
    clear_model_cache()


app = FastAPI(
    title="API de Manutenção Industrial Preditiva (SENAI)",
    description=(
        "Sistema inteligente para estimativa de Vida Útil Restante (RUL - Remaining Useful Life) "
        "de motores e ativos industriais com base no dataset de telemetria da Azure."
    ),
    version="1.0.0",
    lifespan=lifespan
)

# Habilita CORS para permitir conexões de painéis e dashboards
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
def root():
    return {
        "message": "Bem-vindo à API de Manutenção Industrial Preditiva",
        "simulator_url": "/simulador",
        "docs_url": "/docs",
        "health_check": "/health",
        "sqlite_records_url": "/predictions",
        "motors_by_status_url": "/motors/status",
        "email_alerts_url": "/alerts"
    }


@app.get("/simulador", response_class=HTMLResponse, tags=["UI"])
def simulator_ui():
    """Interface Web interativa conectada em tempo real ao modelo ML, registros SQLite, alertas de e-mail e filtro das 3 faixas de RUL."""
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Simulador Web — Manutenção Preditiva (SENAI)</title>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <style>
    :root {
      --bg: #0f172a;
      --card: #1e293b;
      --surface: #0b1120;
      --border: #334155;
      --text: #f8fafc;
      --muted: #94a3b8;
      --primary: #38bdf8;
      --primary-dark: #0284c7;
      --success: #10b981;
      --warning: #f59e0b;
      --danger: #ef4444;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      background: var(--bg);
      color: var(--text);
      padding: 24px;
    }
    .container { max-width: 1280px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
    .header {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 18px 24px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      flex-wrap: wrap;
      gap: 12px;
    }
    .header h1 { font-size: 20px; }
    .header a {
      color: var(--primary);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
      padding: 8px 14px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
    }
    .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .grid { grid-template-columns: 1fr; } }
    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 24px;
      display: flex;
      flex-direction: column;
      gap: 16px;
    }
    .card h2 { font-size: 17px; border-bottom: 1px solid var(--border); padding-bottom: 10px; }
    .form-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    .field { display: flex; flex-direction: column; gap: 6px; }
    .field label { font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; align-items: center; }
    .field label strong { color: var(--text); }
    input[type="number"], input[type="text"], input[type="email"], select {
      padding: 9px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--text);
      font-size: 14px;
    }
    input[type="range"] { width: 100%; accent-color: var(--primary); }
    button.btn-submit {
      padding: 12px 18px;
      border-radius: 10px;
      border: 1px solid var(--primary);
      background: var(--primary-dark);
      color: #fff;
      font-size: 15px;
      font-weight: 700;
      cursor: pointer;
      transition: opacity 0.2s;
    }
    button.btn-submit:hover { opacity: 0.9; }
    .presets { display: flex; gap: 8px; flex-wrap: wrap; }
    .preset-btn {
      padding: 6px 12px;
      border-radius: 8px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      font-size: 12px;
      cursor: pointer;
    }
    .preset-btn:hover { color: var(--text); border-color: var(--primary); }
    .mini-btn {
      padding: 3px 8px;
      border-radius: 6px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--primary);
      font-size: 11px;
      font-weight: 600;
      cursor: pointer;
    }
    .mini-btn:hover { border-color: var(--primary); background: rgba(56, 189, 248, 0.12); }
    .result-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 20px;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .rul-number { font-size: 38px; font-weight: 800; }
    .badge {
      padding: 5px 14px;
      border-radius: 99px;
      font-size: 12px;
      font-weight: 800;
      width: fit-content;
    }
    .filter-tabs { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
    .tab-btn {
      padding: 8px 14px;
      border-radius: 9px;
      border: 1px solid var(--border);
      background: var(--surface);
      color: var(--muted);
      font-size: 12.5px;
      font-weight: 700;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 8px;
    }
    .tab-btn.active {
      border-color: var(--primary);
      color: var(--text);
      background: rgba(56, 189, 248, 0.14);
    }
    .count-pill {
      padding: 2px 8px;
      border-radius: 99px;
      font-size: 11px;
      background: rgba(255,255,255,0.08);
    }
    .alert-banner {
      display: none;
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid var(--warning);
      background: rgba(245, 158, 11, 0.12);
      font-size: 12.5px;
      color: var(--text);
      line-height: 1.45;
    }
    .sqlite-meta-banner {
      display: none;
      padding: 10px 14px;
      border-radius: 10px;
      border: 1px solid var(--border);
      background: rgba(56, 189, 248, 0.08);
      font-size: 12px;
      color: var(--text);
      line-height: 1.45;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); }
    code.uuid-code {
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 11.5px;
      color: var(--primary);
      background: rgba(56, 189, 248, 0.1);
      padding: 2px 6px;
      border-radius: 5px;
    }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--primary); font-weight:700;">Painel de Simulação & Monitoramento PCM em Tempo Real</div>
        <h1>🛠️ Simulador de Vida Útil Restante (RUL) — Motores Industriais</h1>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        <a href="/predictions" target="_blank">🗄️ JSON SQLite (/predictions)</a>
        <a href="/motors/status" target="_blank">📊 JSON Motores (/motors/status)</a>
        <a href="/alerts" target="_blank">📧 JSON Alertas (/alerts)</a>
        <a href="/docs" target="_blank">📄 Swagger (/docs)</a>
        <a href="/health" target="_blank">🩺 Status (/health)</a>
      </div>
    </div>

    <div class="grid">
      <!-- Formulário de Telemetria -->
      <div class="card">
        <h2>1. Parâmetros de Telemetria e Operação</h2>
        <div class="presets">
          <button type="button" class="preset-btn" onclick="applyPreset(1, 168, 445, 98, 38, 5, 0, 4)">🟢 Cenário NORMAL (>168h)</button>
          <button type="button" class="preset-btn" onclick="applyPreset(12, 192, 375, 125, 56, 15, 4, 48)">🟡 Cenário ALERTA (48h–168h)</button>
          <button type="button" class="preset-btn" onclick="applyPreset(25, 215, 310, 145, 72, 20, 8, 75)">🔴 Cenário CRÍTICO (≤48h)</button>
        </div>

        <form id="simForm" onsubmit="runPrediction(event)" style="display:flex; flex-direction:column; gap:14px;">
          <div class="form-grid">
            <div class="field">
              <label>ID do Motor (machine_id)</label>
              <input id="machine_id" type="number" min="1" max="10000" value="1" oninput="onTelemetryChange()" required />
            </div>
            <div class="field">
              <label>Família do Motor (model_type)</label>
              <select id="model_type" onchange="onTelemetryChange()">
                <option value="model1">model1 (Série A)</option>
                <option value="model2">model2 (Série B)</option>
                <option value="model3">model3 (Série C)</option>
                <option value="model4">model4 (Série D)</option>
              </select>
            </div>
            <div class="field">
              <label>Idade do Motor (anos) <strong id="age_lbl">12</strong></label>
              <input id="age" type="range" min="1" max="25" step="1" value="12" oninput="age_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Regime de Trabalho (Horas/Dia)</label>
              <select id="work_shift">
                <option value="24">24 horas (Contínuo)</option>
                <option value="16">16 horas (2 turnos)</option>
                <option value="8">8 horas (1 turno)</option>
              </select>
            </div>
            <div class="field">
              <label>Tensão Elétrica (V) <strong id="volt_lbl">175.0</strong></label>
              <input id="volt" type="range" min="120" max="230" step="0.5" value="175.0" oninput="volt_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Rotação (RPM) <strong id="rotate_lbl">425</strong></label>
              <input id="rotate" type="range" min="250" max="600" step="1" value="425" oninput="rotate_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Pressão (psi) <strong id="pressure_lbl">102.0</strong></label>
              <input id="pressure" type="range" min="65" max="150" step="0.5" value="102.0" oninput="pressure_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Vibração (m/s²) <strong id="vibration_lbl">43.0</strong></label>
              <input id="vibration" type="range" min="20" max="75" step="0.5" value="43.0" oninput="vibration_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Erros nas últimas 24h <strong id="err_lbl">1</strong></label>
              <input id="recent_errors" type="range" min="0" max="10" step="1" value="1" oninput="err_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
            <div class="field">
              <label>Dias desde última manutenção <strong id="maint_lbl">10.5</strong></label>
              <input id="days_since_maint" type="range" min="0" max="90" step="0.5" value="10.5" oninput="maint_lbl.textContent=this.value; onTelemetryChange();" />
            </div>
          </div>

          <div class="form-grid">
            <div class="field">
              <label>E-mail para Alerta Automático (CRÍTICO / ALERTA)</label>
              <input id="alert_email" type="email" value="" placeholder="Deixe vazio para usar ALERT_EMAIL_TO do .env" />
            </div>
            <div class="field">
              <label>
                <span>Idempotency-Key (UUID v4 automático)</span>
                <span style="display:flex; gap:4px;">
                  <button type="button" class="mini-btn" onclick="fillNewUUID()" title="Gerar um novo UUID v4">🎲 Novo UUID</button>
                  <button type="button" class="mini-btn" onclick="clearIdempotencyKey()" title="Limpar campo">🧹 Limpar</button>
                </span>
              </label>
              <input id="idempotency_key" type="text" data-auto-generated="false" oninput="this.dataset.autoGenerated='false'" placeholder="Vazio = gera e preenche UUID v4 ao executar" />
            </div>
          </div>

          <button type="submit" class="btn-submit">⚡ Executar Predição no Modelo (POST /predict)</button>
        </form>
      </div>

      <!-- Resultado da Inferência -->
      <div class="card">
        <h2>2. Resultado do Modelo & Projeção Temporal</h2>
        
        <div class="result-box">
          <div style="display:flex; justify-content:space-between; align-items:center;">
            <span style="font-size:13px; color:var(--muted);">Vida Útil Restante Estimada (RUL)</span>
            <span id="statusBadge" class="badge" style="background:rgba(56,189,248,0.15); color:var(--primary);">AGUARDANDO</span>
          </div>
          <div id="rulDisplay" class="rul-number">-- horas</div>
          <p id="recDisplay" style="font-size:13.5px; color:var(--muted); line-height:1.5;">
            Clique em "Executar Predição no Modelo" para projetar as datas de falha baseadas no modelo <code>models/model.joblib</code>.
          </p>
          <div id="sqliteMetaBanner" class="sqlite-meta-banner"></div>
          <div id="emailAlertBanner" class="alert-banner"></div>
        </div>

        <!-- Container do Gráfico -->
        <div class="result-box" style="padding: 16px;">
          <h3 style="font-size:13px; color:var(--muted); margin-bottom: 10px; font-weight: 500;">Projeção de Degradação (Hoje até Falha Estimada)</h3>
          <div style="position: relative; height: 185px; width: 100%;">
            <canvas id="rulChart"></canvas>
          </div>
        </div>
      </div>
    </div>

    <!-- Seção 3: Visualização das Gravações dos Registros no SQLite (prediction_logs) -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; border-bottom: 1px solid var(--border); padding-bottom: 12px;">
        <div>
          <h2 style="border:none; padding:0;">3. Gravações de Registros no SQLite (Tabela <code>prediction_logs</code> — <code>GET /predictions</code>)</h2>
          <p style="font-size:12px; color:var(--muted); margin-top:4px;">
            Auditoria completa das predições persistidas em <code>data/predictions.db</code> com chave <code>Idempotency-Key</code> (UUID v4). Total exibido: <strong id="sqliteTotalCount" style="color:var(--primary);">0</strong> registros.
          </p>
        </div>
        <div class="filter-tabs">
          <input id="sqliteFilterMachine" type="number" min="1" placeholder="Filtrar Motor #ID" style="width:140px; padding:7px 10px; font-size:12.5px;" oninput="loadSqliteRecords()" />
          <select id="sqliteFilterStatus" onchange="loadSqliteRecords()" style="padding:7px 10px; font-size:12.5px;">
            <option value="">Todos os Status</option>
            <option value="CRÍTICO">🔴 CRÍTICO</option>
            <option value="ALERTA">🟡 ALERTA</option>
            <option value="NORMAL">🟢 NORMAL</option>
          </select>
          <select id="sqliteFilterLimit" onchange="loadSqliteRecords()" style="padding:7px 10px; font-size:12.5px;">
            <option value="20">Últimos 20</option>
            <option value="50" selected>Últimos 50</option>
            <option value="100">Últimos 100</option>
          </select>
          <button type="button" class="tab-btn" onclick="loadSqliteRecords()">🔄 Atualizar SQLite</button>
        </div>
      </div>

      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>ID SQLite</th>
              <th>Data/Hora (Horário Local)</th>
              <th>Idempotency-Key (UUID v4)</th>
              <th>Motor / Família</th>
              <th>Telemetria (V / RPM / psi / m/s²)</th>
              <th>Idade / Erros / Manut.</th>
              <th>RUL Gravada</th>
              <th>Status</th>
              <th>Ação</th>
            </tr>
          </thead>
          <tbody id="sqliteRecordsBody">
            <tr><td colspan="9" style="color:var(--muted);">Carregando gravações do banco SQLite...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Seção 4: Listagem de Motores por Status nas 3 Faixas de RUL -->
    <div class="card">
      <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px; border-bottom: 1px solid var(--border); padding-bottom: 12px;">
        <div>
          <h2 style="border:none; padding:0;">4. Monitoramento de Motores pelas 3 Faixas de RUL (Último Estado por Motor)</h2>
          <p style="font-size:12px; color:var(--muted); margin-top:4px;">Filtre os ativos industriais monitorados de acordo com a criticidade da Vida Útil Restante (<code>GET /motors/status</code>)</p>
        </div>
        <div class="filter-tabs">
          <button type="button" class="tab-btn active" id="tab-ALL" onclick="filterMotorsByBand('')">
            Todos <span class="count-pill" id="cnt-ALL">0</span>
          </button>
          <button type="button" class="tab-btn" id="tab-CRÍTICO" onclick="filterMotorsByBand('CRÍTICO')" style="color:var(--danger);">
            🔴 CRÍTICO (≤ 48h) <span class="count-pill" id="cnt-CRÍTICO">0</span>
          </button>
          <button type="button" class="tab-btn" id="tab-ALERTA" onclick="filterMotorsByBand('ALERTA')" style="color:var(--warning);">
            🟡 ALERTA (48h – 168h) <span class="count-pill" id="cnt-ALERTA">0</span>
          </button>
          <button type="button" class="tab-btn" id="tab-NORMAL" onclick="filterMotorsByBand('NORMAL')" style="color:var(--success);">
            🟢 NORMAL (> 168h) <span class="count-pill" id="cnt-NORMAL">0</span>
          </button>
        </div>
      </div>

      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>Motor</th>
              <th>Família</th>
              <th>Tensão (V)</th>
              <th>Rotação (RPM)</th>
              <th>Vibração (m/s²)</th>
              <th>Erros 24h</th>
              <th>RUL Estimada</th>
              <th>Faixa de Status</th>
              <th>Recomendação Técnica</th>
            </tr>
          </thead>
          <tbody id="motorsBandBody">
            <tr><td colspan="9" style="color:var(--muted);">Carregando motores por faixa de RUL...</td></tr>
          </tbody>
        </table>
      </div>
    </div>

    <!-- Seção 5: Histórico de Alertas de E-mail Disparados -->
    <div class="card">
      <h2>5. Central de Alertas de E-mail Disparados Automaticamente (Tabela SQLite <code>email_alert_logs</code>)</h2>
      <div style="overflow-x:auto;">
        <table>
          <thead>
            <tr>
              <th>ID Alerta</th>
              <th>ID Predição</th>
              <th>Data/Hora (Horário Local)</th>
              <th>Motor / Família</th>
              <th>Status</th>
              <th>RUL (h)</th>
              <th>Destinatário</th>
              <th>Assunto do E-mail</th>
              <th>Modo de Envio</th>
            </tr>
          </thead>
          <tbody id="alertsBody">
            <tr><td colspan="9" style="color:var(--muted);">Nenhum alerta disparado ainda.</td></tr>
          </tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let rulChartInstance = null;
    let currentBandFilter = '';
    let lastProcessedKeys = new Set();

    function generateUUIDv4() {
      if (window.crypto && typeof window.crypto.randomUUID === 'function') {
        return window.crypto.randomUUID();
      }
      return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        const v = c === 'x' ? r : (r & 0x3 | 0x8);
        return v.toString(16);
      });
    }

    function fillNewUUID() {
      const input = document.getElementById('idempotency_key');
      input.value = generateUUIDv4();
      input.dataset.autoGenerated = 'false';
    }

    function clearIdempotencyKey() {
      const input = document.getElementById('idempotency_key');
      input.value = '';
      input.dataset.autoGenerated = 'false';
    }

    function reuseIdempotencyKey(key) {
      const input = document.getElementById('idempotency_key');
      input.value = key;
      input.dataset.autoGenerated = 'false';
      input.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }

    function onTelemetryChange() {
      const input = document.getElementById('idempotency_key');
      if (input && input.dataset.autoGenerated === 'true') {
        input.value = '';
        input.dataset.autoGenerated = 'false';
      }
    }

    function applyPreset(mId, volt, rotate, pressure, vibration, age, err, maint) {
      document.getElementById('machine_id').value = mId;
      document.getElementById('volt').value = volt; document.getElementById('volt_lbl').textContent = volt;
      document.getElementById('rotate').value = rotate; document.getElementById('rotate_lbl').textContent = rotate;
      document.getElementById('pressure').value = pressure; document.getElementById('pressure_lbl').textContent = pressure;
      document.getElementById('vibration').value = vibration; document.getElementById('vibration_lbl').textContent = vibration;
      document.getElementById('age').value = age; document.getElementById('age_lbl').textContent = age;
      document.getElementById('recent_errors').value = err; document.getElementById('err_lbl').textContent = err;
      document.getElementById('days_since_maint').value = maint; document.getElementById('maint_lbl').textContent = maint;
      clearIdempotencyKey();
    }

    function statusColor(st) {
      if (st === 'CRÍTICO') return 'var(--danger)';
      if (st === 'ALERTA') return 'var(--warning)';
      return 'var(--success)';
    }

    function formatLocalTimestamp(ts) {
      if (!ts) return '-';
      try {
        const normalized = (typeof ts === 'string' && !/Z|[+-]\d{2}:?\d{2}$/.test(ts)) ? `${ts}Z` : ts;
        const d = new Date(normalized);
        if (isNaN(d.getTime())) return ts;
        return d.toLocaleString('pt-BR', {
          day: '2-digit',
          month: '2-digit',
          year: 'numeric',
          hour: '2-digit',
          minute: '2-digit',
          second: '2-digit'
        });
      } catch (_) {
        return ts;
      }
    }

    async function loadSqliteRecords() {
      try {
        const machineId = document.getElementById('sqliteFilterMachine').value.trim();
        const statusFilter = document.getElementById('sqliteFilterStatus').value;
        const limit = document.getElementById('sqliteFilterLimit').value || '50';

        const params = new URLSearchParams({ limit });
        if (machineId) params.set('machine_id', machineId);
        if (statusFilter) params.set('status', statusFilter);

        const res = await fetch(`/predictions?${params.toString()}`);
        const logs = await res.json();

        document.getElementById('sqliteTotalCount').textContent = logs.length;
        const tbody = document.getElementById('sqliteRecordsBody');

        if (!logs || !logs.length) {
          tbody.innerHTML = '<tr><td colspan="9" style="color:var(--muted);">Nenhum registro gravado no SQLite para os filtros selecionados.</td></tr>';
          return;
        }

        logs.forEach(l => {
          if (l.idempotency_key) lastProcessedKeys.add(l.idempotency_key);
        });

        tbody.innerHTML = logs.map(l => `
          <tr>
            <td><strong>#${l.id}</strong></td>
            <td style="color:var(--muted); white-space:nowrap;">${formatLocalTimestamp(l.timestamp)}</td>
            <td>
              ${l.idempotency_key
                ? `<code class="uuid-code">${l.idempotency_key}</code>`
                : '<span style="color:var(--muted);">—</span>'}
            </td>
            <td><strong>Motor #${l.machine_id}</strong> (<code>${l.model_type || 'model1'}</code>)</td>
            <td>${l.volt.toFixed(1)}V | ${l.rotate.toFixed(0)}RPM | ${l.pressure.toFixed(1)}psi | ${l.vibration.toFixed(1)}m/s²</td>
            <td>${l.age}a | ${l.recent_errors} err | ${l.days_since_maint}d</td>
            <td><strong>${l.predicted_rul_hours.toFixed(2)}h</strong></td>
            <td><span style="color:${statusColor(l.status)}; font-weight:800;">${l.status}</span></td>
            <td>
              ${l.idempotency_key
                ? `<button type="button" class="mini-btn" onclick="reuseIdempotencyKey('${l.idempotency_key}')">📋 Reutilizar Chave</button>`
                : ''}
            </td>
          </tr>
        `).join('');
      } catch (e) {
        console.error(e);
      }
    }

    async function filterMotorsByBand(band) {
      currentBandFilter = band;
      document.querySelectorAll('.tab-btn').forEach(btn => {
        if (btn.id && btn.id.startsWith('tab-')) btn.classList.remove('active');
      });
      const activeId = band ? `tab-${band}` : 'tab-ALL';
      const activeEl = document.getElementById(activeId);
      if (activeEl) activeEl.classList.add('active');
      await loadMotorsByBand();
    }

    async function loadMotorsByBand() {
      try {
        const url = currentBandFilter
          ? `/motors/status?band=${encodeURIComponent(currentBandFilter)}`
          : '/motors/status';
        const res = await fetch(url);
        const data = await res.json();

        document.getElementById('cnt-ALL').textContent = data.total_motors || 0;
        document.getElementById('cnt-CRÍTICO').textContent = (data.counts_by_band && data.counts_by_band['CRÍTICO']) || 0;
        document.getElementById('cnt-ALERTA').textContent = (data.counts_by_band && data.counts_by_band['ALERTA']) || 0;
        document.getElementById('cnt-NORMAL').textContent = (data.counts_by_band && data.counts_by_band['NORMAL']) || 0;

        const tbody = document.getElementById('motorsBandBody');
        if (!data.motors || !data.motors.length) {
          tbody.innerHTML = '<tr><td colspan="9" style="color:var(--muted);">Nenhum motor encontrado nesta faixa de RUL.</td></tr>';
          return;
        }
        tbody.innerHTML = data.motors.map(m => `
          <tr>
            <td><strong>Motor #${m.machine_id}</strong></td>
            <td><code>${m.model_type || 'model1'}</code></td>
            <td>${m.volt.toFixed(1)} V</td>
            <td>${m.rotate.toFixed(0)} RPM</td>
            <td>${m.vibration.toFixed(1)} m/s²</td>
            <td>${m.recent_errors}</td>
            <td><strong>${m.predicted_rul_hours.toFixed(1)}h</strong></td>
            <td><span style="color:${statusColor(m.status)}; font-weight:800;">${m.status}</span></td>
            <td style="color:var(--muted);">${m.recommendation || ''}</td>
          </tr>
        `).join('');
      } catch (e) {
        console.error(e);
      }
    }

    async function loadEmailAlerts() {
      try {
        const res = await fetch('/alerts?limit=10');
        const alerts = await res.json();
        const tbody = document.getElementById('alertsBody');
        if (!alerts.length) {
          tbody.innerHTML = '<tr><td colspan="9" style="color:var(--muted);">Nenhum alerta de e-mail disparado ainda. Simule um cenário ALERTA ou CRÍTICO para testar.</td></tr>';
          return;
        }
        tbody.innerHTML = alerts.map(a => `
          <tr>
            <td>#${a.id}</td>
            <td>${a.prediction_id ? `#${a.prediction_id}` : '—'}</td>
            <td style="color:var(--muted); white-space:nowrap;">${formatLocalTimestamp(a.timestamp)}</td>
            <td>Motor #${a.machine_id} (<code>${a.model_type || 'model1'}</code>)</td>
            <td><span style="color:${statusColor(a.status)}; font-weight:800;">${a.status}</span></td>
            <td><strong>${a.predicted_rul_hours.toFixed(1)}h</strong></td>
            <td>${a.recipient}</td>
            <td>${a.subject}</td>
            <td><code>${a.delivery_mode}</code></td>
          </tr>
        `).join('');
      } catch (e) {
        console.error(e);
      }
    }

    function drawChart(rulHours, status, shiftHours) {
      shiftHours = shiftHours || 24;
      const factor = 24 / shiftHours;
      const ctx = document.getElementById('rulChart').getContext('2d');
      const now = new Date();
      
      const labels = [];
      const dataPoints = [];
      
      for(let i = 0; i <= 4; i++) {
          const fraction = i / 4;
          const futureDate = new Date(now.getTime() + (rulHours * factor * 3600000 * fraction));
          
          const formatStr = futureDate.getDate().toString().padStart(2, '0') + '/' +
                            (futureDate.getMonth()+1).toString().padStart(2, '0') + ' ' +
                            futureDate.getHours().toString().padStart(2, '0') + ':00';
          labels.push(formatStr);
          dataPoints.push((rulHours * (1 - fraction)).toFixed(1));
      }

      let borderColor = '#10b981';
      let bgColor = 'rgba(16, 185, 129, 0.2)';
      if (status === 'CRÍTICO') {
          borderColor = '#ef4444';
          bgColor = 'rgba(239, 68, 68, 0.2)';
      } else if (status === 'ALERTA') {
          borderColor = '#f59e0b';
          bgColor = 'rgba(245, 158, 11, 0.2)';
      }

      if(rulChartInstance) {
          rulChartInstance.destroy();
      }

      rulChartInstance = new Chart(ctx, {
          type: 'line',
          data: {
              labels: labels,
              datasets: [{
                  label: 'RUL Estimado (h)',
                  data: dataPoints,
                  borderColor: borderColor,
                  backgroundColor: bgColor,
                  borderWidth: 2,
                  fill: true,
                  tension: 0.1,
                  pointBackgroundColor: borderColor,
                  pointBorderColor: '#0f172a'
              }]
          },
          options: {
              responsive: true,
              maintainAspectRatio: false,
              scales: {
                  y: {
                      beginAtZero: true,
                      grid: { color: '#334155' },
                      ticks: { color: '#94a3b8', font: {size: 11} }
                  },
                  x: {
                      grid: { display: false },
                      ticks: { color: '#94a3b8', font: {size: 11}, maxRotation: 0 }
                  }
              },
              plugins: {
                  legend: { display: false },
                  tooltip: {
                      callbacks: {
                          label: function(context) {
                              return context.parsed.y + ' h restantes';
                          }
                      }
                  }
              }
          }
      });
    }

    async function runPrediction(e) {
      e.preventDefault();
      const alertEmail = document.getElementById('alert_email').value.trim();
      const payload = {
        machine_id: parseInt(document.getElementById('machine_id').value, 10),
        model_type: document.getElementById('model_type').value,
        volt: parseFloat(document.getElementById('volt').value),
        rotate: parseFloat(document.getElementById('rotate').value),
        pressure: parseFloat(document.getElementById('pressure').value),
        vibration: parseFloat(document.getElementById('vibration').value),
        age: parseFloat(document.getElementById('age').value),
        recent_errors: parseInt(document.getElementById('recent_errors').value, 10),
        days_since_maint: parseFloat(document.getElementById('days_since_maint').value),
        alert_email: alertEmail || null
      };

      // Gera e preenche automaticamente um UUID v4 caso o campo Idempotency-Key esteja vazio
      const idemInput = document.getElementById('idempotency_key');
      let idemKey = idemInput.value.trim();
      if (!idemKey) {
        idemKey = generateUUIDv4();
        idemInput.value = idemKey;
        idemInput.dataset.autoGenerated = 'true';
      }

      const wasAlreadyProcessed = lastProcessedKeys.has(idemKey);

      const headers = {
        'Content-Type': 'application/json',
        'Idempotency-Key': idemKey
      };

      const res = await fetch('/predict', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      });
      const data = await res.json();

      if (data.idempotency_key) {
        idemInput.value = data.idempotency_key;
        lastProcessedKeys.add(data.idempotency_key);
      }

      document.getElementById('rulDisplay').textContent = `${data.predicted_rul_hours.toFixed(2)} horas`;
      document.getElementById('recDisplay').textContent = data.recommendation;

      const badge = document.getElementById('statusBadge');
      badge.textContent = data.status;
      if (data.status === 'CRÍTICO') {
        badge.style.background = 'rgba(239,68,68,0.2)';
        badge.style.color = 'var(--danger)';
      } else if (data.status === 'ALERTA') {
        badge.style.background = 'rgba(245,158,11,0.2)';
        badge.style.color = 'var(--warning)';
      } else {
        badge.style.background = 'rgba(16,185,129,0.2)';
        badge.style.color = 'var(--success)';
      }

      const sqliteMetaBanner = document.getElementById('sqliteMetaBanner');
      sqliteMetaBanner.style.display = 'block';
      sqliteMetaBanner.innerHTML = wasAlreadyProcessed
        ? `♻️ <strong>Idempotência Ativa (Sem Duplicação no SQLite):</strong> Registro SQLite <strong>#${data.record_id || '—'}</strong> retornado para a chave <code class="uuid-code">${data.idempotency_key || idemKey}</code>.`
        : `🗄️ <strong>Gravado no SQLite:</strong> Registro <strong>#${data.record_id || '—'}</strong> | <code>Idempotency-Key</code>: <code class="uuid-code">${data.idempotency_key || idemKey}</code>`;

      const alertBanner = document.getElementById('emailAlertBanner');
      if (data.email_alert_triggered) {
        alertBanner.style.display = 'block';
        alertBanner.style.borderColor = data.status === 'CRÍTICO' ? 'var(--danger)' : 'var(--warning)';
        alertBanner.innerHTML = `📧 <strong>Alerta Automático de E-mail Disparado!</strong> Destinatário: <code>${data.email_recipient || alertEmail}</code> | Modo: <code>${data.email_delivery_mode}</code>`;
      } else {
        alertBanner.style.display = 'none';
      }

      const shiftH = parseFloat(document.getElementById("work_shift").value) || 24;
      drawChart(data.predicted_rul_hours, data.status, shiftH);

      await loadSqliteRecords();
      await loadMotorsByBand();
      await loadEmailAlerts();
    }

    loadSqliteRecords();
    loadMotorsByBand();
    loadEmailAlerts();
  </script>
</body>
</html>
"""


app.include_router(health.router)
app.include_router(predict.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
