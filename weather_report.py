import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Weather Report Analysis", layout="wide")

# ── Animated rainy weather background ─────────────────────────────────────────
st.markdown("""
<style>
/* 🌧️ Animated Rainy Weather Background */
.stApp {
  background: linear-gradient(180deg, #0a1628 0%, #1a2b4a 50%, #0d2137 100%);
  overflow: hidden;
}

/* Container for all weather elements */
.weather-bg {
  position: fixed;
  top: 0; left: 0;
  width: 100vw; height: 100vh;
  pointer-events: none;
  z-index: 0;
  overflow: hidden;
}

/* ☁️ Cloud icons moving across the sky — transparent */
.cloud {
  position: absolute;
  width: 100px;
  height: 60px;
  background: rgba(255,255,255,0.25); /* semi-transparent white */
  border-radius: 50px;
  box-shadow:
    30px 0 0 0 rgba(255,255,255,0.25),
    60px 0 0 0 rgba(255,255,255,0.25),
    90px 0 0 0 rgba(255,255,255,0.25),
    45px -20px 0 0 rgba(255,255,255,0.25),
    75px -20px 0 0 rgba(255,255,255,0.25);
  opacity: 0.4; /* lighter opacity for text visibility */
  filter: blur(1px);
  animation: moveCloud linear infinite;
}
@keyframes moveCloud {
  from { transform: translateX(-200px); opacity: 0.3; }
  50%  { opacity: 0.5; }
  to   { transform: translateX(110vw); opacity: 0.3; }
}

/* Different cloud positions and speeds */
.c1 { top: 10%; animation-duration: 40s; }
.c2 { top: 25%; animation-duration: 55s; animation-delay: -10s; }
.c3 { top: 40%; animation-duration: 70s; animation-delay: -20s; }
.c4 { top: 60%; animation-duration: 50s; animation-delay: -5s; }

/* 🌧️ Falling raindrops */
.drop {
  position: absolute;
  width: 2px;
  height: 15px;
  background: rgba(147,210,255,0.6);
  border-radius: 2px;
  animation: rainFall linear infinite;
}
@keyframes rainFall {
  from { top: -20px; opacity: 0.9; }
  to   { top: 110vh; opacity: 0.1; }
}

/* ⚡ Lightning flashes */
.lightning {
  position: absolute;
  top: 0;
  left: 70%;
  width: 5px;
  height: 100vh;
  background: rgba(255,255,255,0.6); /* slightly dimmer lightning */
  opacity: 0;
  animation: flashLightning 6s infinite;
}
@keyframes flashLightning {
  0%, 95%, 100% { opacity: 0; }
  96% { opacity: 0.8; }
  97% { opacity: 0.4; }
  98% { opacity: 0.8; }
  99% { opacity: 0; }
}
</style>

<div class="weather-bg">
  <div class="cloud c1"></div>
  <div class="cloud c2"></div>
  <div class="cloud c3"></div>
  <div class="cloud c4"></div>

  <!-- Raindrops -->
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>
  <div class="drop"></div><div class="drop"></div><div class="drop"></div>

  <!-- Lightning -->
  <div class="lightning"></div>
</div>
""", unsafe_allow_html=True)

# ── Title ─────────────────────────────────────────────────────────────────────
st.title("🌧️ Weather Report Analysis")
st.markdown("**6 Years of Daily Weather Data (2020–2025) · 8 Major Indian Cities**")

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.header("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["Dashboard", "Temperature Analysis", "Humidity Analysis",
     "Rainfall Analysis", "Statistics", "🔮 Predictions"]
)

# (Keep your existing data, ML, and visualization logic unchanged below)
# ── Constants ─────────────────────────────────────────────────────────────────
cities = ["Mumbai", "Delhi", "Bangalore", "Chennai",
          "Kolkata", "Hyderabad", "Pune", "Jaipur"]

CITY_CONFIG = {
    "Mumbai":    {"base_temp": 28, "base_hum": 78},
    "Delhi":     {"base_temp": 25, "base_hum": 58},
    "Bangalore": {"base_temp": 23, "base_hum": 65},
    "Chennai":   {"base_temp": 30, "base_hum": 75},
    "Kolkata":   {"base_temp": 27, "base_hum": 72},
    "Hyderabad": {"base_temp": 27, "base_hum": 60},
    "Pune":      {"base_temp": 25, "base_hum": 62},
    "Jaipur":    {"base_temp": 26, "base_hum": 48},
}

MONTH_LABELS = ["Jan","Feb","Mar","Apr","May","Jun",
                "Jul","Aug","Sep","Oct","Nov","Dec"]
T = "plotly_dark"

# ── Data ──────────────────────────────────────────────────────────────────────
@st.cache_data
def build_dataframe():
    np.random.seed(42)
    date_range = pd.date_range(start="2020-01-01", end="2025-12-31", freq="D")
    records = []
    for city in cities:
        cfg   = CITY_CONFIG[city]
        n     = len(date_range)
        doy   = date_range.dayofyear
        month = date_range.month

        temp = (cfg["base_temp"]
                + 8 * np.sin(2 * np.pi * (doy - 80) / 365)
                + np.random.normal(0, 2, n))

        hum = np.clip(
            cfg["base_hum"]
            + 18 * np.sin(2 * np.pi * (doy - 170) / 365)
            + np.random.normal(0, 5, n),
            30, 100
        )

        monsoon = np.where((month >= 6) & (month <= 9), 1.0, 0.08)
        rain    = np.clip(np.random.exponential(6, n) * monsoon, 0, 120)

        for i, d in enumerate(date_range):
            records.append({
                "Date":        d,
                "City":        city,
                "Temperature": round(float(temp[i]), 2),
                "Humidity":    round(float(hum[i]), 2),
                "Rainfall":    round(float(rain[i]), 2),
            })

    df = pd.DataFrame(records)
    df["Year"]      = df["Date"].dt.year
    df["Month"]     = df["Date"].dt.month
    df["Day"]       = df["Date"].dt.day
    df["DayOfYear"] = df["Date"].dt.dayofyear
    return df

df = build_dataframe()

# ── ML — no DataFrame argument passed to avoid st.cache_resource hashing bug ─
@st.cache_resource
def train_models():
    _df = build_dataframe()

    le    = LabelEncoder()
    df_ml = _df.copy()
    df_ml["CityEncoded"] = le.fit_transform(df_ml["City"])

    features = ["Year", "Month", "Day", "DayOfYear", "CityEncoded"]
    models, scores = {}, {}

    for target in ["Temperature", "Humidity", "Rainfall"]:
        X = df_ml[features]
        y = df_ml[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        model = RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores[target] = {
            "MAE": round(mean_absolute_error(y_test, y_pred), 2),
            "R2":  round(r2_score(y_test, y_pred), 3),
        }
        models[target] = model

    return models, le, scores

models, le, model_scores = train_models()

# ════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ════════════════════════════════════════════════════════════════════════════
if page == "Dashboard":
    st.header("📊 Dashboard Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{len(df):,}")
    c2.metric("Cities Analyzed",  len(cities))
    c3.metric("Years of Data",    6)
    c4.metric("Overall Avg Temp", f"{df['Temperature'].mean():.1f} °C")

    st.divider()

    st.subheader("Monthly Average Temperature — All Cities")
    df_m = df.groupby(["City", "Year", "Month"])["Temperature"].mean().reset_index()
    df_m["Period"] = (df_m["Year"].astype(str) + "-"
                      + df_m["Month"].astype(str).str.zfill(2))
    fig = px.line(df_m.sort_values("Period"), x="Period", y="Temperature",
                  color="City", template=T,
                  title="Monthly Avg Temperature Trend (2020-2025)")
    fig.update_layout(height=420, xaxis_title="Month",
                      yaxis_title="Temp (°C)", hovermode="x unified")
    fig.update_xaxes(nticks=24)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("City Snapshot")
    snap = df.groupby("City").agg(
        Avg_Temp=("Temperature","mean"),
        Avg_Humidity=("Humidity","mean"),
        Total_Rainfall=("Rainfall","sum")
    ).round(1).reset_index()
    fig2 = px.scatter(snap, x="Avg_Temp", y="Avg_Humidity",
                      size="Total_Rainfall", color="City", template=T,
                      title="Avg Temp vs Humidity (bubble size = total rainfall)",
                      labels={"Avg_Temp":"Avg Temp (°C)",
                              "Avg_Humidity":"Avg Humidity (%)"},
                      size_max=55)
    fig2.update_layout(height=420)
    st.plotly_chart(fig2, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TEMPERATURE ANALYSIS — distribution histogram removed
# ════════════════════════════════════════════════════════════════════════════
elif page == "Temperature Analysis":
    st.header("🌡️ Temperature Analysis")

    city_filter = st.multiselect("Filter Cities", cities, default=cities)
    dff = df[df["City"].isin(city_filter)]

    avg = dff.groupby("City")["Temperature"].mean().sort_values().reset_index()
    fig = px.bar(avg, x="Temperature", y="City", orientation="h",
                 color="Temperature", color_continuous_scale="Reds",
                 template=T, title="Average Temperature by City",
                 text=avg["Temperature"].round(1))
    fig.update_traces(texttemplate="%{text} °C", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.box(dff, x="City", y="Temperature", color="City",
                 template=T, title="Temperature Spread per City", points="outliers")
    fig.update_layout(height=440, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    monthly = dff.groupby(["City","Month"])["Temperature"].mean().reset_index()
    fig = px.line(monthly, x="Month", y="Temperature", color="City",
                  markers=True, template=T,
                  title="Seasonal Temperature Pattern (monthly avg)")
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_LABELS)
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# HUMIDITY ANALYSIS — distribution histogram removed
# ════════════════════════════════════════════════════════════════════════════
elif page == "Humidity Analysis":
    st.header("💧 Humidity Analysis")

    city_filter = st.multiselect("Filter Cities", cities, default=cities)
    dff = df[df["City"].isin(city_filter)]

    avg = dff.groupby("City")["Humidity"].mean().sort_values().reset_index()
    fig = px.bar(avg, x="Humidity", y="City", orientation="h",
                 color="Humidity", color_continuous_scale="Blues",
                 template=T, title="Average Humidity by City",
                 text=avg["Humidity"].round(1))
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    fig = px.violin(dff, x="City", y="Humidity", color="City",
                    box=True, points="outliers",
                    template=T, title="Humidity Violin Plot — Full Distribution")
    fig.update_layout(height=460, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    monthly = dff.groupby(["City","Month"])["Humidity"].mean().reset_index()
    fig = px.line(monthly, x="Month", y="Humidity", color="City",
                  markers=True, template=T,
                  title="Seasonal Humidity Pattern (monsoon peak Jun-Sep)")
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_LABELS)
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# RAINFALL ANALYSIS — distribution fixed + heatmap colours fixed
# ════════════════════════════════════════════════════════════════════════════
elif page == "Rainfall Analysis":
    st.header("🌧️ Rainfall Analysis")

    city_filter = st.multiselect("Filter Cities", cities, default=cities)
    dff = df[df["City"].isin(city_filter)]

    total = (dff.groupby("City")["Rainfall"].sum()
               .sort_values(ascending=False).reset_index())
    fig = px.bar(total, x="City", y="Rainfall",
                 color="Rainfall", color_continuous_scale="Blues",
                 template=T, title="Total Rainfall by City (6 Years)",
                 text=total["Rainfall"].round(0))
    fig.update_traces(texttemplate="%{text} mm", textposition="outside")
    fig.update_layout(height=420, coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

    # ── Rainfall distribution — faceted, one panel per city ──────────────────
    st.subheader("Rainfall Distribution per City (rainy days only)")
    rain_only = dff[dff["Rainfall"] > 0].copy()
    bins   = [0, 5, 15, 30, 50, 80, 120]
    labels = ["0-5 mm","5-15 mm","15-30 mm","30-50 mm","50-80 mm","80-120 mm"]
    rain_only["Range"] = pd.cut(rain_only["Rainfall"], bins=bins,
                                labels=labels, right=True)
    rain_binned = (rain_only.groupby(["City","Range"], observed=True)
                            .size().reset_index(name="Days"))
    fig = px.bar(rain_binned, x="Range", y="Days", color="Range",
                 facet_col="City", facet_col_wrap=4,
                 color_discrete_sequence=px.colors.sequential.Blues[2:],
                 template=T,
                 title="Rainfall Intensity Distribution — Each City Separately",
                 labels={"Range":"Rainfall (mm)", "Days":"# Days"})
    fig.update_layout(height=500, showlegend=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
    st.plotly_chart(fig, use_container_width=True)

    # ── Monthly Rainfall Heatmap — warm scale + cell values shown ────────────
    st.subheader("Monthly Rainfall Heatmap")
    heat  = dff.groupby(["City","Month"])["Rainfall"].mean().reset_index()
    pivot = heat.pivot(index="City", columns="Month", values="Rainfall").round(1)
    pivot.columns = MONTH_LABELS
    fig = px.imshow(
        pivot,
        color_continuous_scale="YlOrRd",
        zmin=0,
        zmax=float(pivot.values.max()),
        aspect="auto",
        text_auto=True,
        template=T,
        title="Avg Daily Rainfall (mm) — Monsoon peak clearly visible",
        labels={"color":"Rainfall (mm)"}
    )
    fig.update_traces(textfont_size=11)
    fig.update_layout(height=420, coloraxis_colorbar=dict(title="mm"))
    st.plotly_chart(fig, use_container_width=True)

    monthly = dff.groupby(["City","Month"])["Rainfall"].mean().reset_index()
    fig = px.bar(monthly, x="Month", y="Rainfall", color="City",
                 barmode="group", template=T,
                 title="Monthly Avg Rainfall per City")
    fig.update_xaxes(tickvals=list(range(1,13)), ticktext=MONTH_LABELS)
    fig.update_layout(height=420, hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# STATISTICS
# ════════════════════════════════════════════════════════════════════════════
elif page == "Statistics":
    st.header("📈 Weather Statistics")

    for label, metric in [("🌡️ Temperature", "Temperature"),
                           ("💧 Humidity",    "Humidity"),
                           ("🌧️ Rainfall",   "Rainfall")]:
        st.subheader(f"{label} Statistics")
        stats = df.groupby("City")[metric].agg(["min","max","mean","std"]).round(2)
        stats.columns = ["Min","Max","Mean","Std Dev"]
        st.dataframe(stats.style.background_gradient(cmap="Blues", axis=0),
                     use_container_width=True)
        st.divider()

# ════════════════════════════════════════════════════════════════════════════
# PREDICTIONS — fixed by removing df argument from train_models()
# ════════════════════════════════════════════════════════════════════════════
elif page == "🔮 Predictions":
    st.header("🔮 Weather Predictions")
    st.markdown(
        "**Random Forest Regressor** trained on 6 years of data · "
        "Features: year, month, day, day-of-year, city"
    )

    st.subheader("Model Accuracy (20% hold-out test set)")
    c1, c2, c3 = st.columns(3)
    for col, m in zip([c1, c2, c3], ["Temperature", "Humidity", "Rainfall"]):
        col.metric(f"{m} MAE", f"{model_scores[m]['MAE']}")
        col.metric(f"{m} R²",  f"{model_scores[m]['R2']}")

    st.divider()

    c1, c2, c3 = st.columns(3)
    with c1:
        selected_city = st.selectbox("City", cities, key="pred_city")
    with c2:
        metric = st.selectbox("Metric to Predict",
                              ["Temperature", "Humidity", "Rainfall"],
                              key="pred_metric")
    with c3:
        days_ahead = st.slider("Days to Predict", 7, 365, 90, step=7,
                               key="pred_days")

    unit_map = {"Temperature": "°C", "Humidity": "%", "Rainfall": "mm"}
    unit = unit_map[metric]

    last_date    = df["Date"].max()
    future_dates = pd.date_range(
        start=last_date + timedelta(days=1), periods=days_ahead, freq="D"
    )

    city_encoded = int(le.transform([selected_city])[0])
    future_X = pd.DataFrame({
        "Year":        future_dates.year.astype(int),
        "Month":       future_dates.month.astype(int),
        "Day":         future_dates.day.astype(int),
        "DayOfYear":   future_dates.dayofyear.astype(int),
        "CityEncoded": city_encoded,
    })

    predictions = models[metric].predict(future_X)
    hist_std    = float(df[df["City"] == selected_city][metric].std())
    upper       = predictions + hist_std
    lower       = np.maximum(predictions - hist_std, 0)

    hist = (df[(df["City"] == selected_city) & (df["Year"] == 2025)]
              .tail(180)[["Date", metric]].copy())

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=hist["Date"], y=hist[metric],
        name="Historical (last 6 months)", mode="lines",
        line=dict(color="#60a5fa", width=1.8)
    ))
    fig.add_trace(go.Scatter(
        x=list(future_dates) + list(future_dates[::-1]),
        y=list(upper) + list(lower[::-1]),
        fill="toself", fillcolor="rgba(251,191,36,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        name="Confidence Band (±1σ)", hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=future_dates, y=predictions,
        name="Predicted", mode="lines",
        line=dict(color="#f59e0b", width=2.5, dash="dash")
    ))
    fig.update_layout(
        template="plotly_dark",
        title=f"{metric} Forecast — {selected_city} · Next {days_ahead} Days",
        xaxis_title="Date",
        yaxis_title=f"{metric} ({unit})",
        height=520,
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.18)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Forecast Summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Predicted Avg", f"{predictions.mean():.1f} {unit}")
    c2.metric("Predicted Max", f"{predictions.max():.1f} {unit}")
    c3.metric("Predicted Min", f"{predictions.min():.1f} {unit}")
    c4.metric("Std Dev",       f"{predictions.std():.1f} {unit}")

    pred_df = pd.DataFrame({
        "Date":                  future_dates.strftime("%Y-%m-%d"),
        f"Predicted_{metric}":   predictions.round(2),
        "Upper_Bound":           upper.round(2),
        "Lower_Bound":           lower.round(2),
    })
    with st.expander("📋 View Forecast Table"):
        st.dataframe(pred_df, use_container_width=True)

    st.download_button(
        "📥 Download Predictions CSV",
        pred_df.to_csv(index=False),
        f"{selected_city}_{metric}_{days_ahead}d_forecast.csv",
        "text/csv"
    )

st.markdown("---")
