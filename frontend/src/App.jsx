import { useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import "./App.css";

const API = "http://127.0.0.1:8000";

function App() {
  const [stats, setStats] = useState(null);
  const [models, setModels] = useState([]);
  const [forecast, setForecast] = useState([]);
  const [city, setCity] = useState("chennai");
  const [shap, setShap] = useState([]);

  useEffect(() => {
    Promise.all([
  fetch(`${API}/api/site-stats?city=${city}`).then((r) => r.json()),
  fetch(`${API}/api/models`).then((r) => r.json()),
  fetch(`${API}/api/forecast?city=${city}`).then((r) => r.json()),
  fetch(`${API}/api/shap?city=${city}`).then((r) => r.json()),
])
  .then(([siteData, modelData, forecastData, shapData]) => {
    setStats(siteData);
    setModels(modelData);
    setForecast(forecastData);
    setShap(shapData);
  })
  .catch((err) => console.error("API Error:", err));
  }, [city]);

  if (!stats) {
    return (
      <div className="loading">
        <div className="loader"></div>
        Loading Renewable Energy Analytics...
      </div>
    );
  }

  const cityModels = models.filter(
    (m) => m.City.toLowerCase() === city
  );

  const bestModel = cityModels.reduce(
    (best, current) =>
      !best || current.MAE_MW < best.MAE_MW ? current : best,
    null
  );

  return (
    <div className="app">

      {/* SIDEBAR */}
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">⚡</div>
          <div>
            <h2>RENEW-X</h2>
            <span>AI Energy Intelligence</span>
          </div>
        </div>

        <nav>
  <button
    className="nav-item active"
    onClick={() =>
      document.getElementById("dashboard")?.scrollIntoView({
        behavior: "smooth",
      })
    }
  >
    <span>◉</span>
    Dashboard
  </button>

  <button
    className="nav-item"
    onClick={() =>
      document.getElementById("forecasting")?.scrollIntoView({
        behavior: "smooth",
      })
    }
  >
    <span>◈</span>
    Forecasting
  </button>

  <button
    className="nav-item"
    onClick={() =>
      document.getElementById("validation")?.scrollIntoView({
        behavior: "smooth",
      })
    }
  >
    <span>◫</span>
    Model Validation
  </button>

  <button
    className="nav-item"
    onClick={() =>
      document.getElementById("explainability")?.scrollIntoView({
        behavior: "smooth",
      })
    }
  >
    <span>◌</span>
    Explainability
  </button>
</nav>

        <div className="sidebar-bottom">
          <div className="status-dot"></div>
          <div>
            <strong>System Online</strong>
            <small>FastAPI + ML Pipeline</small>
          </div>
        </div>
      </aside>

      {/* MAIN */}
      <main id="dashboard" className="main">

        {/* TOP BAR */}
        <header className="topbar">
          <div>
            <div className="eyebrow">RENEWABLE ENERGY ANALYTICS</div>
            <h1>Hybrid Energy Intelligence</h1>
            <p>
              Explainable AI framework for renewable power forecasting
            </p>
          </div>

          <div className="controls">
            <label>LOCATION</label>
            <select
              value={city}
              onChange={(e) => setCity(e.target.value)}
            >
              <option value="chennai">Chennai</option>
              <option value="delhi">Delhi</option>
            </select>
          </div>
        </header>

        {/* HERO */}
        <section className="hero">
          <div>
            <span className="hero-label">HYBRID RENEWABLE SITE</span>
            <h2>1 km² Optimized Energy Site</h2>
            <p>
              Solar + wind generation optimized through machine learning
              forecasting.
            </p>
          </div>

          <div className="hero-capacity">
            <span>Total Capacity</span>
            <strong>{stats.hybrid_capacity_mw} MW</strong>
          </div>
        </section>

        {/* KPI CARDS */}
        <section className="kpi-grid">

          <div className="kpi">
            <div className="kpi-icon solar">☀</div>
            <div>
              <span>Solar Capacity</span>
              <strong>{stats.solar_capacity_mw} MW</strong>
              <small>60% land allocation</small>
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-icon wind">♨</div>
            <div>
              <span>Wind Capacity</span>
              <strong>{stats.wind_capacity_mw} MW</strong>
              <small>40% land allocation</small>
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-icon energy">⚡</div>
            <div>
              <span>Annual Generation</span>
              <strong>
                {(stats.annual_total_generation_mwh / 1000).toFixed(2)}
                <small className="unit"> GWh</small>
              </strong>
              <small>Estimated annual output</small>
            </div>
          </div>

          <div className="kpi">
            <div className="kpi-icon roi">↗</div>
            <div>
              <span>25-Year ROI</span>
              <strong>{stats.roi_25_year_percent.toFixed(1)}%</strong>
              <small>{stats.payback_years.toFixed(1)} year payback</small>
            </div>
          </div>

        </section>

        {/* MIDDLE GRID */}
        <section className="dashboard-grid">

          {/* LAND ALLOCATION */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <span className="section-label">SITE UTILIZATION</span>
                <h3>Land Allocation</h3>
              </div>
              <span className="badge">1 km²</span>
            </div>

            <div className="land-content">

              <div
                className="donut"
                style={{
                  background:
                    "conic-gradient(#f5b942 0deg 216deg, #4b8df8 216deg 360deg)",
                }}
              >
                <div className="donut-inner">
                  <strong>1.0</strong>
                  <span>km²</span>
                </div>
              </div>

              <div className="legend">

                <div className="legend-item">
                  <span className="legend-dot solar-dot"></span>
                  <div>
                    <strong>Solar</strong>
                    <span>0.60 km² · 60%</span>
                  </div>
                </div>

                <div className="legend-item">
                  <span className="legend-dot wind-dot"></span>
                  <div>
                    <strong>Wind</strong>
                    <span>0.40 km² · 40%</span>
                  </div>
                </div>

              </div>
            </div>
          </div>

          {/* FINANCIAL */}
          <div className="panel">
            <div className="panel-header">
              <div>
                <span className="section-label">PROJECT ECONOMICS</span>
                <h3>Financial Overview</h3>
              </div>
            </div>

            <div className="finance-list">

              <div>
                <span>Total CAPEX</span>
                <strong>
                  ₹{(stats.total_capex_inr / 10000000).toFixed(2)} Cr
                </strong>
              </div>

              <div>
                <span>Annual Revenue</span>
                <strong>
                  ₹{(stats.annual_revenue_inr / 10000000).toFixed(2)} Cr
                </strong>
              </div>

              <div>
                <span>Annual Net Revenue</span>
                <strong>
                  ₹{(stats.annual_net_revenue_inr / 10000000).toFixed(2)} Cr
                </strong>
              </div>

              <div className="payback">
                <span>Payback Period</span>
                <strong>{stats.payback_years.toFixed(2)} years</strong>
              </div>

            </div>
          </div>

        </section>

                {/* FORECAST CHART */}
        <section id="forecasting" className="panel forecast-panel">
          <div className="panel-header">
            <div>
              <span className="section-label">POWER FORECAST</span>
              <h3>Actual vs Predicted Power</h3>
              <p>2025 out-of-time validation · First 200 hourly observations</p>
            </div>

            <div className="best-model">
              <span>MODEL</span>
              <strong>XGBoost</strong>
            </div>
          </div>

          <div className="forecast-chart">
            <ResponsiveContainer width="100%" height={360}>
              <LineChart
                data={forecast.map((item, index) => ({
  ...item,
  index: index + 1,
}))}
                margin={{ top: 10, right: 20, left: 10, bottom: 10 }}
              >
                <CartesianGrid strokeDasharray="3 3" />

                <XAxis
                  dataKey="index"
                  tick={false}
                  axisLine={false}
                />

                <YAxis
                  label={{
                    value: "Power (MW)",
                    angle: -90,
                    position: "insideLeft",
                  }}
                />

                <Tooltip />

                <Legend />

                <Line
                  type="monotone"
                  dataKey="Actual_Power_MW"
                  name="Actual Power"
                  stroke="#4b8df8"
                  strokeWidth={2}
                  dot={false}
                />

                <Line
                  type="monotone"
                  dataKey="Predicted_Power_MW"
                  name="Predicted Power"
                  stroke="#f5b942"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
        {/* MODEL PERFORMANCE */}
        <section id="validation" className="panel model-panel">

          <div className="panel-header">
            <div>
              <span className="section-label">MACHINE LEARNING</span>
              <h3>Model Performance</h3>
              <p>2025 Out-of-Time validation</p>
            </div>

            <div className="best-model">
              <span>BEST MODEL</span>
              <strong>XGBoost</strong>
            </div>
          </div>

          <div className="model-list">

            {cityModels.map((model) => {

              const score = Math.max(
                5,
                Math.min(100, 100 - model.MAE_MW * 25)
              );

              return (
                <div className="model-row" key={model.Model}>

                  <div className="model-name">
                    <strong>{model.Model}</strong>
                    <span>MAE {model.MAE_MW.toFixed(3)} MW</span>
                  </div>

                  <div className="bar-container">
                    <div
                      className={`model-bar ${
                        model.Model === "XGBOOST"
                          ? "best-bar"
                          : ""
                      }`}
                      style={{ width: `${score}%` }}
                    ></div>
                  </div>

                  <div className="model-r2">
                    <strong>{model.R2.toFixed(4)}</strong>
                    <span>R²</span>
                  </div>

                </div>
              );
            })}

          </div>
        </section>

                {/* EXPLAINABILITY */}
        <section id="explainability" className="panel shap-panel">
          <div className="panel-header">
            <div>
              <span className="section-label">EXPLAINABLE AI</span>
              <h3>SHAP Feature Importance</h3>
              <p>
                Top features influencing the XGBoost forecasting model
              </p>
            </div>

            <div className="best-model">
              <span>METHOD</span>
              <strong>SHAP</strong>
            </div>
          </div>

          <div className="shap-list">
  {shap
    .slice()
    .sort((a, b) => b.Mean_Absolute_SHAP - a.Mean_Absolute_SHAP)
    .map((item, index) => {
      const maxValue = shap[0]?.Mean_Absolute_SHAP || 1;
      const width = (item.Mean_Absolute_SHAP / maxValue) * 100;

      return (
        <div className="shap-row" key={item.Feature}>
          <div className="shap-rank">
            {index + 1}
          </div>

          <div className="shap-name">
            <strong>{item.Feature}</strong>
          </div>

          <div className="shap-bar-container">
            <div
              className="shap-bar"
              style={{ width: `${width}%` }}
            ></div>
          </div>

          <div className="shap-value">
            {item.Mean_Absolute_SHAP.toFixed(3)}
          </div>
        </div>
      );
    })}
</div>

<div className="shap-insight">
  <div className="shap-insight-icon">💡</div>

  <div>
    <strong>What does this tell us?</strong>

    <p>
      GHI (Global Horizontal Irradiance) is the dominant feature influencing
      the forecast, with a mean absolute SHAP value of 9.105. Wind speed is
      the second most influential feature, while 24-hour lagged power and
      lagged GHI capture temporal patterns from previous observations.
    </p>
  </div>
</div>
        </section>

        {/* VALIDATION HIGHLIGHT */}
        {bestModel && (
          <section className="validation-card">

            <div className="validation-icon">✓</div>

            <div className="validation-text">
              <span>VALIDATION RESULT</span>

              <h3>
                XGBoost achieved the strongest forecasting performance
              </h3>

              <p>
                On the {city === "chennai" ? "Chennai" : "Delhi"} 2025
                out-of-time dataset, XGBoost achieved an R² of{" "}
                <strong>{bestModel.R2.toFixed(4)}</strong> with an MAE of{" "}
                <strong>{bestModel.MAE_MW.toFixed(3)} MW</strong>.
              </p>
            </div>

            <div className="validation-metric">
              <strong>{bestModel.R2.toFixed(4)}</strong>
              <span>R² Score</span>
            </div>

          </section>
        )}

        <footer>
          Hybrid Explainable AI Framework · Renewable Energy Forecasting
        </footer>

      </main>
    </div>
  );
}

export default App;