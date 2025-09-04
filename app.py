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
num_simulations = st.sidebar.slider("Número de simulaciones", 10, 500, 100, step=10)
days_ahead = st.sidebar.slider("Días a futuro", 30, 730, 365, step=30)

# Permite múltiples precios objetivo separados por coma
price_targets_input = st.sidebar.text_input("🎯 Precio(s) objetivo (USD, separados por coma)", "100000,150000,200000")
price_targets = [float(p.strip()) for p in price_targets_input.split(",") if p.strip().isdigit()]

method = st.sidebar.selectbox("Método de clasificación", ["Desviación estándar", "Percentiles"])

# --- Descargar datos ---
@st.cache_data
def load_data():
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start="2021-01-01", end=today, interval="1d", auto_adjust=True)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(0)
    btc['Change'] = btc['Close'].pct_change()
    return btc

btc_data = load_data()
btc_data = btc_data.dropna(subset=['Change'])

# --- Clasificación ---
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

# --- Matriz de transición y retornos ---
states = btc_data['State'].astype(int).values
changes = btc_data['Change'].values

def transition_matrix(states):
    trans = np.zeros((3, 3))
    for (curr, nxt) in zip(states[:-1], states[1:]):
        trans[curr, nxt] += 1
    return trans / trans.sum(axis=1, keepdims=True)

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

# --- Percentiles ---
p10 = sim_df.quantile(0.10, axis=1)
p25 = sim_df.quantile(0.25, axis=1)
p50 = sim_df.quantile(0.50, axis=1)
p75 = sim_df.quantile(0.75, axis=1)
p90 = sim_df.quantile(0.90, axis=1)

# --- Probabilidad para cada precio objetivo ---
st.subheader("🎯 Probabilidades de superar precios objetivo")
final_prices = sim_df.iloc[-1]
for pt in price_targets:
    prob = (final_prices > pt).mean() * 100
    st.write(f"📌 Probabilidad de superar **${pt:,.0f}**: **{prob:.2f}%**")
# --- Explicación de los percentiles y probabilidad ---
st.markdown(
    f"""
### 🧠 ¿Cómo interpretar estos resultados?

- **P10**: Solo el 10% de las simulaciones dieron precios **más bajos** que este valor. Es un escenario pesimista.
- **P25**: El 25% de los precios simulados fueron más bajos. Es un escenario moderadamente pesimista.
- **P50 (Mediana)**: Es el valor central. La mitad de las simulaciones resultaron por **encima** y la otra mitad **por debajo** de este precio.
- **P75**: El 75% de los precios simulados fueron menores. Solo el 25% superaron este valor, por lo tanto es un escenario optimista.
- **P90**: Solo el 10% de las simulaciones superaron este precio. Es un escenario muy optimista.

---

### 🎯 Precio objetivo

Se calculó la **probabilidad de que el precio de Bitcoin supere los ${price_target:,.0f} USD** en los próximos **{days_ahead} días**, usando **{num_simulations} simulaciones** basadas en un modelo de **cadenas de Markov**.

🔮 **Probabilidad de superar ${price_target:,.0f}: `{prob_over:.2f}%`**
    """,
    unsafe_allow_html=False
)


Este porcentaje indica cuántas simulaciones terminaron con un precio **superior** al objetivo que ingresaste.
""")


# --- Gráfico ---
st.subheader("📉 Simulación de precios futuros de BTC")

fig, ax = plt.subplots(figsize=(14, 7))

# Gráfico con diferentes colores
ax.plot(p10, color='red', linestyle='--', label='P10')
ax.plot(p25, color='orange', linestyle='--', label='P25')
ax.plot(p50, color='blue', linewidth=2, label='Mediana (P50)')
ax.plot(p75, color='green', linestyle='--', label='P75')
ax.plot(p90, color='purple', linestyle='--', label='P90')
ax.fill_between(sim_df.index, p10, p90, alpha=0.1, color='gray', label='Rango P10–P90')

# Ejes más legibles
ax.set_xticks(np.arange(0, days_ahead + 1, 15))
min_price = sim_df.min().min()
max_price = sim_df.max().max()
yticks = np.arange(int(min_price // 10000) * 10000, int(max_price // 10000 + 2) * 10000, 10000)
ax.set_yticks(yticks)

ax.set_xlabel("Día")
ax.set_ylabel("Precio (USD)")
ax.set_title("Simulación Monte Carlo para Bitcoin")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# --- Tabla de precios simulados al último día ---
st.subheader("📊 Precio simulado al final del período")

last_day = {
    'P10': p10.iloc[-1],
    'P25': p25.iloc[-1],
    'P50': p50.iloc[-1],
    'P75': p75.iloc[-1],
    'P90': p90.iloc[-1],
}

last_day_df = pd.DataFrame.from_dict(last_day, orient='index', columns=['Precio simulado (USD)'])
last_day_df.index.name = "Percentil"
st.dataframe(last_day_df.style.format("${:,.0f}"))

# --- Descargar CSV ---
st.download_button(
    label="⬇️ Descargar resultados CSV",
    data=sim_df.to_csv().encode('utf-8'),
    file_name=f"simulaciones_btc_{method.lower()}.csv",
    mime='text/csv'
)



