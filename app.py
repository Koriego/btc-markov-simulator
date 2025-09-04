@st.cache_data
def load_data(start_date):
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start=start_date, end=today, interval="1d")
    st.write("Columnas descargadas:", btc.columns.tolist())
    if btc.empty:
        st.error("No se descargaron datos.")
        st.stop()
    if 'Close' not in btc.columns and 'Adj Close' not in btc.columns:
        st.error("No hay columnas 'Close' ni 'Adj Close'.")
        st.stop()
    # Usar 'Adj Close' si existe, si no 'Close'
    price_col = 'Adj Close' if 'Adj Close' in btc.columns else 'Close'
    btc['Change'] = btc[price_col].pct_change()
    return btc
