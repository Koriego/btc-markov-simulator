import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import matplotlib.pyplot as plt

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Simulación de Bitcoin", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔧 Configuración")
start_dates = {
    "2021-01-01": "2021-01-01",
    "2022-01-01": "2022-01-01",
    "2023-01-01": "2023-01-01",
    "2024-01-01": "2024-01-01",
}
selected_start = st.sidebar.selectbox("Fecha inicio de datos", list(start_dates.keys()), index=0)
num_simulations = st.sidebar.slider("Número de simulaciones", 10, 500, 100, step=10)
days_ahead = st.sidebar.selectbox("Días a futuro", [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 360, 730], index=2)
price_target = st.sidebar.number_input("🎯 Precio objetivo (USD)", value=100000)
method = st.sidebar.selectbox("Método de clasificación", ["Desviación estándar", "Percentiles"])

@st.cache
def load_data(start_date):
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start=start_date, end=today, interval="1d")
    # Debug columnas
    st.write("Columnas descargadas:", btc.columns.tolist())
    if btc.empty:
        st.error("No se descargaron datos.")
        st.stop()
    if 'Close' not in btc.columns and 'Adj Close' not in btc.columns:
        st.error("No hay columnas 'Close' ni 'Adj Close'.")
        st.stop()
    price_col = 'Adj Close' if 'Adj Close' in btc.columns else 'Close'
    btc['Change'] = btc[price_col].pct_change()
    return btc

btc_data = load_data(start_dates[selected_start])

if 'Change' in btc_data.columns:
    btc_data.dropna(subset=['Change'], inplace=True)
else:
    st.error("La columna 'Change' no existe en los datos descargados.")
    st.stop()

# --- Clasificaciones ---
def classify_std(change, mean, std):
    if change < mean - std:
        return 0
    elif change > mean + std:
        return 2
    else:
        return 1

def classify_percentile(change, low, high):
    if change < low:
        return 0
    elif change > high:
        return 2
    else:
        return 1

mean_change = btc_data['Change'].mean()
std_change = btc_data['Change'].std()
lower_th = np.percentile(btc_data['Change'], 33)
upper_th = np.percentile(btc_data['Change'], 66)

if method == "Desviación estándar":
    btc_data['State'] = btc_data['Change'].apply(lambda x: classify_std(x, mean_change, std_change))
else:
    btc_data['State'] = btc_data['Change'].apply(lambda x: classify_percentile(x, lower_th, upper_th))

# --- Matriz de transición ---
states = btc_data['State'].astype(int).values
changes = btc_data['Change'].values

def transition_matrix(states):
    trans = np.zeros((3, 3))
    for (curr, nxt) in zip(states[:-1], states[1:]):
        trans[curr, nxt] += 1
    # Evitar división por cero
    with np.errstate(invalid='ignore'):
        trans = np.divide(trans, trans.sum(axis=1, keepdims=True), out=np.zeros_like(trans), where=trans.sum(axis=1, keepdims=True)!=0)
    return trans

def avg_return_by_state(states, changes):
    df = pd.DataFrame({'State': states, 'Change': changes})
    return df.groupby('State')['Change'].mean().to_dict()

T = transition_matrix(states)
R = avg_return_by_state(states, changes)
last_state = states[-1]
last_price = btc_data.iloc[-1]['Adj Close'] if 'Adj Close' in btc_data.columns else btc_data.iloc[-1]['Close']

# --- Mostrar matrices ---
st.subheader("📊 Matriz de transición")
state_labels = ["📉 Baja", "➖ Estable", "📈 Sube"]
st.dataframe(pd.DataFrame(T, columns=state_labels, index=state_labels).round(3))

st.subheader("📈 Promedio de cambio por estado")
for k, v in R.items():
    st.write(f"Estado {k} ({state_labels[k]}): {v:.4f}")

# --- Simulación ---
def simulate(start_state, steps, matrix, returns):
    states = [start_state]
    for _ in range(steps):
        current = states[-1]
        next_state = np.random.choice([0, 1, 2], p=matrix[current])
        states.append(next_state)
    return states

sim_df = pd.DataFrame()
for i in range(num_simulations):
    s_states = simulate(last_state, days_ahead, T, R)
    prices = [last_price]
    for s in s_states[1:]:
        prices.append(prices[-1] * (1 + R[s]))
    sim_df[f'Sim_{i+1}'] = prices

# --- Estadísticas ---
p10 = sim_df.quantile(0.10, axis=1)
p25 = sim_df.quantile(0.25, axis=1)
p50 = sim_df.quantile(0.50, axis=1)
p75 = sim_df.quantile(0.75, axis=1)
p90 = sim_df.quantile(0.90, axis=1)

final_prices = sim_df.iloc[-1]
prob_over = (final_prices > price_target).mean() * 100

st.subheader("📈 Simulación de precios")
st.write(f"🎯 Probabilidad de superar ${price_target:,.0f}: **{prob_over:.2f}%**")

fig, ax = plt.subplots(figsize=(12, 6))
ax.fill_between(sim_df.index, p10, p90, alpha=0.2, label='P10–P90')
ax.plot(p50, label="Mediana (P50)", color='blue', linewidth=2)
ax.plot(p25, '--', color='gray', alpha=0.5, label='P25 / P75')
ax.plot(p75, '--', color='gray', alpha=0.5)
ax.set_xlabel("Día")
ax.set_ylabel("Precio (USD)")
ax.set_title("Simulación de Bitcoin")
ax.legend()
ax.grid(True)
st.pyplot(fig)

# --- Descargar CSV ---
st.download_button(
    label="⬇️ Descargar resultados CSV",
    data=sim_df.to_csv().encode('utf-8'),
    file_name=f"simulaciones_btc_{method.lower()}.csv",
    mime='text/csv'
)
