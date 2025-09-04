import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import plotly.graph_objects as go

# --- CONFIGURACIÓN ---
st.set_page_config(page_title="Simulación de Bitcoin", layout="wide")

# --- Sidebar ---
st.sidebar.title("🔧 Configuración")
num_simulations = st.sidebar.slider("Número de simulaciones", 10, 500, 100, step=10)
days_ahead = st.sidebar.slider("Días a futuro", 30, 730, 365, step=30)
price_target = st.sidebar.number_input("🎯 Precio objetivo (USD)", value=100000)
method = st.sidebar.selectbox("Método de clasificación", ["Desviación estándar", "Percentiles"])

# --- Descargar datos ---
@st.cache_data
def load_data():
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start="2021-01-01", end=today, interval="1d", auto_adjust=True)
    
    # Corregir multi-índice en columnas si existe
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)

    btc['Change'] = btc['Close'].pct_change()
    btc.dropna(subset=['Change'], inplace=True)
    return btc

btc_data = load_data()

# Validación: si no hay datos suficientes, mostrar error y detener app
if btc_data.empty or btc_data['Change'].empty:
    st.error("No hay suficientes datos para realizar la simulación. Verifica la conexión y la fuente de datos.")
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
    with np.errstate(divide='ignore', invalid='ignore'):
        trans = np.nan_to_num(trans / trans.sum(axis=1, keepdims=True))
    return trans

def avg_return_by_state(states, changes):
    df = pd.DataFrame({'State': states, 'Change': changes})
    return df.groupby('State')['Change'].mean().to_dict()

T = transition_matrix(states)
R = avg_return_by_state(states, changes)
last_state = states[-1]
last_price = btc_data['Close'].iloc[-1]

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

# --- Gráfica interactiva con Plotly ---
fig = go.Figure()

# Rango de días (índice)
x = sim_df.index

# Añadir área entre p10 y p90
fig.add_trace(go.Scatter(
    x=x.tolist() + x[::-1].tolist(),
    y=p90.tolist() + p10[::-1].tolist(),
    fill='toself',
    fillcolor='rgba(0,100,80,0.2)',
    line=dict(color='rgba(255,255,255,0)'),
    hoverinfo="skip",
    showlegend=True,
    name='P10–P90',
))

# Añadir líneas p25, p50, p75
fig.add_trace(go.Scatter(
    x=x, y=p50,
    mode='lines',
    line=dict(color='blue', width=2),
    name='Mediana (P50)'
))

fig.add_trace(go.Scatter(
    x=x, y=p25,
    mode='lines',
    line=dict(color='gray', width=1, dash='dash'),
    name='P25'
))

fig.add_trace(go.Scatter(
    x=x, y=p75,
    mode='lines',
    line=dict(color='gray', width=1, dash='dash'),
    name='P75'
))

# Configurar ticks eje X cada 15 días
fig.update_xaxes(
    dtick=15,
    title_text='Día',
    showgrid=True
)

# Configurar ticks eje Y cada 10000 USD
min_price = sim_df.min().min()
max_price = sim_df.max().max()
start_ytick = int(np.floor(min_price / 10000) * 10000)
end_ytick = int(np.ceil(max_price / 10000) * 10000)
fig.update_yaxes(
    dtick=10000,
    range=[start_ytick, end_ytick],
    title_text='Precio (USD)',
    showgrid=True,
    tickformat=',d'
)

fig.update_layout(
    title="Simulación de Bitcoin",
    hovermode="x unified",
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
)

st.plotly_chart(fig, use_container_width=True)

# --- Descargar CSV ---
st.download_button(
    label="⬇️ Descargar resultados CSV",
    data=sim_df.to_csv().encode('utf-8'),
    file_name=f"simulaciones_btc_{method.lower()}.csv",
    mime='text/csv'
)
