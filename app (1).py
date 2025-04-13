import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from prophet import Prophet

st.title("Time Series Forecasting APP")

# Upload a CSV
uploaded_file = st.file_uploader("Upload a CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file, parse_dates=['InvoiceDate'])
    df = df.groupby('InvoiceDate')['Quantity'].sum().reset_index()
    df = df[df['Quantity'] > 0]
    st.info("Note: Records with non-positive Quantity (0 or negative) have been removed for time series analysis.")
    df.set_index('InvoiceDate', inplace=True)

    st.write("Raw Time Series")
    st.line_chart(df)  
    decomposition_type = st.selectbox("Choose Decomposition Type", ["Additive", "Multiplicative"])
    result = seasonal_decompose(df, model=decomposition_type.lower(), period=30)

    st.write("Decomposition Results")
    st.pyplot(result.plot())

from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import numpy as np

# Choose Forecasting Model
model_choice = st.selectbox("Choose a forecasting model", ["ARIMA", "ETS", "Prophet"])
st.write(f"You selected: {model_choice}")

forecast_days = 30  # Forecast horizon

if model_choice == "Prophet":
    st.subheader("Prophet Forecast")
    prophet_df = df.reset_index()
    prophet_df.columns = ['ds', 'y']
    m = Prophet()
    m.fit(prophet_df)
    future = m.make_future_dataframe(periods=forecast_days)
    forecast = m.predict(future)

    fig1 = m.plot(forecast)
    st.pyplot(fig1)
    fig2 = m.plot_components(forecast)
    st.pyplot(fig2)

    # Evaluation (only if we have enough history)
    if len(df) > forecast_days:
        actual = df['Quantity'][-forecast_days:]
        predicted = forecast['yhat'][-forecast_days:].values
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2%}")

elif model_choice == "ARIMA":
    st.subheader("ARIMA Forecast")
    from pmdarima import auto_arima
    from statsmodels.tsa.arima.model import ARIMA

    stepwise_model = auto_arima(df['Quantity'], seasonal=False, trace=True)
    best_order = stepwise_model.order
    st.write(f"Best ARIMA order: {best_order}")

    model = ARIMA(df['Quantity'], order=best_order)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=forecast_days)

    last_date = df.index[-1]
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast.index = future_index

    st.line_chart(forecast)

    if len(df) > forecast_days:
        actual = df['Quantity'][-forecast_days:]
        predicted = model_fit.predict(start=len(df)-forecast_days, end=len(df)-1)
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2%}")

elif model_choice == "ETS":
    st.subheader("ETS Forecast")
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    model = ExponentialSmoothing(df['Quantity'], trend="add", seasonal="add", seasonal_periods=30)
    model_fit = model.fit()
    forecast = model_fit.forecast(forecast_days)

    last_date = df.index[-1]
    future_index = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=forecast_days, freq='D')
    forecast.index = future_index

    st.line_chart(forecast)

    if len(df) > forecast_days:
        actual = df['Quantity'][-forecast_days:]
        predicted = model_fit.fittedvalues[-forecast_days:]
        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mape = mean_absolute_percentage_error(actual, predicted)
        st.write(f"RMSE: {rmse:.2f}")
        st.write(f"MAPE: {mape:.2%}")
