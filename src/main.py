from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from src.routers import health, predict
from src.database.connection import engine, Base
from src.services.ml_service import load_model_on_startup, clear_model_cache


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Fase 3 (Ciclo de Vida e Inicialização Otimizada - Lifespan):
    Inicializa as tabelas do banco SQLite e carrega o artefato serializado (models/model.joblib)
    uma única vez na memória RAM durante o startup do servidor ASGI.
    """
    Base.metadata.create_all(bind=engine)
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
        "health_check": "/health"
    }


@app.get("/simulador", response_class=HTMLResponse, tags=["UI"])
def simulator_ui():
    """Interface Web interativa conectada em tempo real ao modelo ML e ao banco SQLite, incluindo gráfico de projeção temporal."""
    return """
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Simulador Web — Manutenção Preditiva (SENAI)</title>
  <!-- Injeção do Chart.js para o gráfico de projeção -->
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
    .container { max-width: 1200px; margin: 0 auto; display: flex; flex-direction: column; gap: 20px; }
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
    .field label { font-size: 12px; color: var(--muted); display: flex; justify-content: space-between; }
    .field label strong { color: var(--text); }
    input[type="number"], input[type="text"], select {
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
    .result-box {
      background: var(--surface);
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 22px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }
    .rul-number { font-size: 40px; font-weight: 800; }
    .badge {
      padding: 5px 14px;
      border-radius: 99px;
      font-size: 12px;
      font-weight: 800;
      width: fit-content;
    }
    table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
    th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid var(--border); }
    th { color: var(--muted); }
  </style>
</head>
<body>
  <div class="container">
    <div class="header">
      <div>
        <div style="font-size:11px; text-transform:uppercase; letter-spacing:0.06em; color:var(--primary); font-weight:700;">Painel de Simulação em Tempo Real</div>
        <h1>🛠️ Simulador de Vida Útil Restante (RUL) — Motores Industriais</h1>
      </div>
      <div style="display:flex; gap:10px;">
        <a href="/docs" target="_blank">📄 Abrir Swagger (/docs)</a>
        <a href="/health" target="_blank">🩺 Status (/health)</a>
      </div>
    </div>

    <div class="grid">
      <!-- Formulário de Telemetria -->
      <div class="card">
        <h2>1. Parâmetros de Telemetria e Operação</h2>
        <div class="presets">
          <button type="button" class="preset-btn" onclick="applyPreset(1, 168, 445, 98, 38, 5, 0, 4)">🟢 Cenário Saudável</button>
          <button type="button" class="preset-btn" onclick="applyPreset(12, 182, 410, 114, 48, 14, 2, 28)">🟡 Cenário de Desgaste</button>
          <button type="button" class="preset-btn" onclick="applyPreset(25, 198, 360, 128, 62, 19, 5, 55)">🔴 Cenário Severo</button>
        </div>

        <form id="simForm" onsubmit="runPrediction(event)" style="display:flex; flex-direction:column; gap:16px;">
          <div class="form-grid">
            <div class="field">
              <label>ID do Motor (machine_id)</label>
              <input id="machine_id" type="number" min="1" max="10000" value="1" required />
            </div>
            <div class="field">
              <label>Família do Motor (model_type)</label>
              <select id="model_type">
                <option value="model1">model1 (Série A)</option>
                <option value="model2">model2 (Série B)</option>
                <option value="model3">model3 (Série C)</option>
                <option value="model4">model4 (Série D)</option>
              </select>
            </div>
            <div class="field">
              <label>Idade do Motor (anos) <strong id="age_lbl">12</strong></label>
              <input id="age" type="range" min="1" max="25" step="1" value="12" oninput="age_lbl.textContent=this.value" />
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
              <input id="volt" type="range" min="120" max="230" step="0.5" value="175.0" oninput="volt_lbl.textContent=this.value" />
            </div>
            <div class="field">
              <label>Rotação (RPM) <strong id="rotate_lbl">425</strong></label>
              <input id="rotate" type="range" min="250" max="600" step="1" value="425" oninput="rotate_lbl.textContent=this.value" />
            </div>
            <div class="field">
              <label>Pressão (psi) <strong id="pressure_lbl">102.0</strong></label>
              <input id="pressure" type="range" min="65" max="150" step="0.5" value="102.0" oninput="pressure_lbl.textContent=this.value" />
            </div>
            <div class="field">
              <label>Vibração (m/s²) <strong id="vibration_lbl">43.0</strong></label>
              <input id="vibration" type="range" min="20" max="75" step="0.5" value="43.0" oninput="vibration_lbl.textContent=this.value" />
            </div>
            <div class="field">
              <label>Erros nas últimas 24h <strong id="err_lbl">1</strong></label>
              <input id="recent_errors" type="range" min="0" max="10" step="1" value="1" oninput="err_lbl.textContent=this.value" />
            </div>
            <div class="field">
              <label>Dias desde última manutenção <strong id="maint_lbl">10.5</strong></label>
              <input id="days_since_maint" type="range" min="0" max="90" step="0.5" value="10.5" oninput="maint_lbl.textContent=this.value" />
            </div>
          </div>

          <div class="field">
            <label>Idempotency-Key (opcional — evita duplicação em retentativas)</label>
            <input id="idempotency_key" type="text" placeholder="Deixe em branco para gerar novo log a cada clique" />
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
        </div>

        <!-- Container do Gráfico -->
        <div class="result-box" style="margin-top: 4px; padding: 16px;">
          <h3 style="font-size:13px; color:var(--muted); margin-bottom: 12px; font-weight: 500;">Projeção de Degradação (Hoje até Falha Estimada)</h3>
          <div style="position: relative; height: 180px; width: 100%;">
            <canvas id="rulChart"></canvas>
          </div>
        </div>

        <h2 style="margin-top: 8px;">3. Últimas Predições Salvas no SQLite</h2>
        <div style="overflow-x:auto;">
          <table>
            <thead>
              <tr>
                <th>ID</th>
                <th>Motor</th>
                <th>Vibração</th>
                <th>Erros 24h</th>
                <th>RUL (h)</th>
                <th>Status</th>
              </tr>
            </thead>
            <tbody id="historyBody">
              <tr><td colspan="6" style="color:var(--muted);">Carregando histórico...</td></tr>
            </tbody>
          </table>
        </div>
      </div>
    </div>
  </div>

  <script>
    let rulChartInstance = null;

    function applyPreset(mId, volt, rotate, pressure, vibration, age, err, maint) {
      document.getElementById('machine_id').value = mId;
      document.getElementById('volt').value = volt; document.getElementById('volt_lbl').textContent = volt;
      document.getElementById('rotate').value = rotate; document.getElementById('rotate_lbl').textContent = rotate;
      document.getElementById('pressure').value = pressure; document.getElementById('pressure_lbl').textContent = pressure;
      document.getElementById('vibration').value = vibration; document.getElementById('vibration_lbl').textContent = vibration;
      document.getElementById('age').value = age; document.getElementById('age_lbl').textContent = age;
      document.getElementById('recent_errors').value = err; document.getElementById('err_lbl').textContent = err;
      document.getElementById('days_since_maint').value = maint; document.getElementById('maint_lbl').textContent = maint;
    }

    async function loadHistory() {
      try {
        const res = await fetch('/predictions?limit=4');
        const logs = await res.json();
        const tbody = document.getElementById('historyBody');
        if (!logs.length) {
          tbody.innerHTML = '<tr><td colspan="6" style="color:var(--muted);">Nenhum registro ainda.</td></tr>';
          return;
        }
        tbody.innerHTML = logs.map(l => `
          <tr>
            <td>#${l.id}</td>
            <td>Motor ${l.machine_id}</td>
            <td>${l.vibration} m/s²</td>
            <td>${l.recent_errors}</td>
            <td><strong>${l.predicted_rul_hours.toFixed(1)}h</strong></td>
            <td>${l.status}</td>
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
      const payload = {
        machine_id: parseInt(document.getElementById('machine_id').value, 10),
        model_type: document.getElementById('model_type').value,
        volt: parseFloat(document.getElementById('volt').value),
        rotate: parseFloat(document.getElementById('rotate').value),
        pressure: parseFloat(document.getElementById('pressure').value),
        vibration: parseFloat(document.getElementById('vibration').value),
        age: parseFloat(document.getElementById('age').value),
        recent_errors: parseInt(document.getElementById('recent_errors').value, 10),
        days_since_maint: parseFloat(document.getElementById('days_since_maint').value)
      };

      const headers = { 'Content-Type': 'application/json' };
      const idemKey = document.getElementById('idempotency_key').value.trim();
      if (idemKey) headers['Idempotency-Key'] = idemKey;

      const res = await fetch('/predict', {
        method: 'POST',
        headers,
        body: JSON.stringify(payload)
      });
      const data = await res.json();

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

      const shiftH = parseFloat(document.getElementById("work_shift").value) || 24;
      drawChart(data.predicted_rul_hours, data.status, shiftH);

      loadHistory();
    }

    loadHistory();
  </script>
</body>
</html>
"""


app.include_router(health.router)
app.include_router(predict.router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)
