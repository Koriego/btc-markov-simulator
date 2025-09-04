@st.cache_data
def load_data(start_date):
    today = datetime.today().strftime('%Y-%m-%d')
    btc = yf.download("BTC-USD", start=start_date, end=today, interval="1d", auto_adjust=True)
    
    # Validar que no esté vacío
    if btc.empty:
        st.error("No se pudieron descargar datos. Intenta con otra fecha de inicio o revisa conexión.")
        st.stop()

    # Si las columnas tienen MultiIndex, aplanar (obtener último nivel)
    if isinstance(btc.columns, pd.MultiIndex):
        btc.columns = btc.columns.get_level_values(-1)

    # Verificar que exista la columna 'Close'
    if 'Close' not in btc.columns:
        st.error("La columna 'Close' no está en los datos descargados.")
        st.stop()

    btc['Change'] = btc['Close'].pct_change()
    return btc


