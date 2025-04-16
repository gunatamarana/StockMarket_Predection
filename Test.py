import streamlit as st
import yaml
import bcrypt
import os
from yaml.loader import SafeLoader
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
import re
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import plotly.express as px  # Import Plotly for interactive charts

# ---------------- Helper Functions (No changes needed) ----------------
def load_config(path="credentials.yaml"):
    if os.path.exists(path):
        with open(path, "r") as file:
            try:
                return yaml.safe_load(file)
            except yaml.YAMLError:
                st.error("❌ YAML loading failed.")
                return None
    else:
        return {
            "credentials": {"usernames": {}},
            "cookie": {"expiry_days": 30, "key": "abcdef", "name": "stock_app"}
        }

def save_config(config, path="credentials.yaml"):
    with open(path, "w") as file:
        yaml.dump(config, file, default_flow_style=False)

def fetch_stock_data(stock_symbol, data_period="1y"):
    return yf.download(stock_symbol, period=data_period)

def fetch_live_price(stock_symbol):
    stock = yf.Ticker(stock_symbol)
    live_price = stock.history(period="1d")["Close"].iloc[0]
    return live_price

def is_strong_password(password):
    return (len(password) >= 8 and
            re.search(r"[A-Z]", password) and
            re.search(r"[a-z]", password) and
            re.search(r"[0-9]", password) and
            re.search(r"[\W_]", password))

# ---------------- Enhanced Model (No changes needed) ----------------
def enhanced_model(stock_data):
    df = stock_data.copy()
    for col in ['Open', 'High', 'Low', 'Close']:
        df[f'{col}_SMA_5'] = df[col].rolling(5).mean()
        df[f'{col}_Lag_1'] = df[col].shift(1)
        df[f'{col}_Lag_2'] = df[col].shift(2)

    df.dropna(inplace=True)

    X = df[[
        'Open_SMA_5', 'High_SMA_5', 'Low_SMA_5', 'Close_SMA_5',
        'Open_Lag_1', 'Open_Lag_2',
        'High_Lag_1', 'High_Lag_2',
        'Low_Lag_1', 'Low_Lag_2',
        'Close_Lag_1', 'Close_Lag_2',
        'Volume'
    ]]
    y_cols = ['Open', 'High', 'Low', 'Close']
    y = df[y_cols]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred, multioutput='uniform_average')
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # Forecast next 7 days
    future_predictions = []
    last_rows = df.tail(10).copy() # Use .copy() to avoid SettingWithCopyWarning
    for i in range(7):
        features = {
            'Open_SMA_5': last_rows['Open'].tail(5).mean(),
            'High_SMA_5': last_rows['High'].tail(5).mean(),
            'Low_SMA_5': last_rows['Low'].tail(5).mean(),
            'Close_SMA_5': last_rows['Close'].tail(5).mean(),
            'Open_Lag_1': last_rows.iloc[-1]['Open'],
            'Open_Lag_2': last_rows.iloc[-2]['Open'],
            'High_Lag_1': last_rows.iloc[-1]['High'],
            'High_Lag_2': last_rows.iloc[-2]['High'],
            'Low_Lag_1': last_rows.iloc[-1]['Low'],
            'Low_Lag_2': last_rows.iloc[-2]['Low'],
            'Close_Lag_1': last_rows.iloc[-1]['Close'],
            'Close_Lag_2': last_rows.iloc[-2]['Close'],
            'Volume': last_rows.iloc[-1]['Volume']
        }
        X_future = pd.DataFrame([features])
        y_future = model.predict(X_future)[0]
        prediction = dict(zip(y_cols, y_future))
        future_predictions.append(prediction)

        new_row = pd.DataFrame([{
            'Open': prediction['Open'],
            'High': prediction['High'],
            'Low': prediction['Low'],
            'Close': prediction['Close'],
            'Volume': last_rows.iloc[-1]['Volume']
        }])
        last_rows = pd.concat([last_rows, new_row], ignore_index=True).tail(10)
        df = pd.concat([df, new_row], ignore_index=True)


    future_dates = [date.today() + timedelta(days=i) for i in range(1, 8)]
    pred_df = pd.DataFrame(future_predictions)
    pred_df["Date"] = future_dates

    return pred_df, r2, mae, mse, rmse

# ---------------- Register & Login (No changes needed) ----------------
def register_user():
    st.subheader("🔏 Register New User")
    new_name = st.text_input("Full Name")
    new_username = st.text_input("Username")
    new_password = st.text_input("Password", type="password")
    confirm_password = st.text_input("Confirm Password", type="password")

    if new_password and not is_strong_password(new_password):
        st.warning("Weak password. Use uppercase, number, symbol.")

    if st.button("Register"):
        if not all([new_name, new_username, new_password, confirm_password]):
            st.warning("⚠️ Fill all fields.")
            return
        if new_password != confirm_password:
            st.error("❌ Passwords do not match.")
            return
        if not is_strong_password(new_password):
            st.error("❌ Weak password format.")
            return

        config = load_config()
        if new_username in config["credentials"]["usernames"]:
            st.error("❌ Username already exists.")
            return

        hashed_password = bcrypt.hashpw(new_password.encode(), bcrypt.gensalt()).decode()
        config["credentials"]["usernames"][new_username] = {
            "name": new_name,
            "password": hashed_password
        }

        save_config(config)
        st.success("✅ Registered! Now login.")

def login_user():
    st.subheader("🔑 Login to your account")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if not username or not password:
            st.warning("⚠️ Fill in both fields.")
            return

        config = load_config()
        if config is None or username not in config["credentials"]["usernames"]:
            st.error("❌ Username not found.")
            return

        stored_password = config["credentials"]["usernames"][username]["password"]
        if bcrypt.checkpw(password.encode(), stored_password.encode()):
            st.session_state["authenticated"] = True
            st.session_state["username"] = username
            st.session_state["name"] = config["credentials"]["usernames"][username]["name"]
            st.success(f"✅ Welcome back, {st.session_state['name']}!")
        else:
            st.error("❌ Incorrect password.")

# ---------------- Main App with Time Range Selection Above Charts ----------------
def main():
    st.set_page_config(page_title="📈 Stock Prediction App", layout="wide")

    if st.session_state.get("authenticated"):
        with st.sidebar:
            st.markdown(f"👋 Welcome, **{st.session_state['username']}**")
            st.title("📍 Select Stock")
            stock_exchange = st.selectbox("Stock Exchange", ["Indian Stock Market", "US Stock Market"])
            default_symbol = "RELIANCE.NS" if stock_exchange == "Indian Stock Market" else "AAPL"
            stock_symbol = st.text_input("Enter Stock Symbol", value=default_symbol)

            st.title("⏳ Data Period for Prediction")
            prediction_data_period_options = ["3mo", "6mo", "1y", "3y", "5y"]
            prediction_data_period = st.selectbox("Select Data Period for Prediction", prediction_data_period_options, index=2) # Default to 1 year

            if st.button("🚪 Logout"):
                st.session_state.clear()
                st.rerun()

        if stock_symbol:
            stock_data = fetch_stock_data(stock_symbol, prediction_data_period)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(1) # Remove the secondary level (stock symbol)
            currency_symbol = "₹" if stock_exchange == "Indian Stock Market" else "$"
            live_price = fetch_live_price(stock_symbol)

            st.subheader(f"🔴 Live Price for {stock_symbol}: {currency_symbol}{live_price:.2f}")

            # Display historical data as a dataframe
            st.subheader(f"📅 Last 5 Days Data for {stock_symbol}")
            st.dataframe(stock_data.tail())

            # --- Historical Data Visualization ---
            st.subheader("📊 Historical Data Visualization")
            time_range = st.radio("Select Time Range for Charts",
                                  ["1 Month", "3 Months", "6 Months", "1 Year", "3 Years", "5 Years", "All Data"],
                                  index=3, # Default to 1 Year
                                  horizontal=True)

            today = date.today()
            if time_range == "1 Month":
                start_date = today - pd.DateOffset(months=1)
                plot_data = stock_data[stock_data.index >= start_date]
            elif time_range == "3 Months":
                start_date = today - pd.DateOffset(months=3)
                plot_data = stock_data[stock_data.index >= start_date]
            elif time_range == "6 Months":
                start_date = today - pd.DateOffset(months=6)
                plot_data = stock_data[stock_data.index >= start_date]
            elif time_range == "1 Year":
                start_date = today - pd.DateOffset(years=1)
                plot_data = stock_data[stock_data.index >= start_date]
            elif time_range == "3 Years":
                start_date = today - pd.DateOffset(years=3)
                plot_data = stock_data[stock_data.index >= start_date]
            elif time_range == "5 Years":
                start_date = today - pd.DateOffset(years=5)
                plot_data = stock_data[stock_data.index >= start_date]
            else: # "All Data"
                plot_data = stock_data

            if not plot_data.empty:
                # Line chart for Close price
                st.subheader("Close Price")
                st.line_chart(plot_data["Close"])

                # Area chart for Volume
                st.subheader("Volume")
                st.area_chart(plot_data["Volume"])

                # Interactive chart using Plotly for Open, High, Low, Close
                st.subheader("Open, High, Low, Close Prices (Interactive)")
                fig = px.line(plot_data, x=plot_data.index, y=["Open", "High", "Low", "Close"],
                              title=f"Interactive OHLC Chart for {stock_symbol} ({time_range})")
                st.plotly_chart(fig)
            else:
                st.warning(f"No data available for the selected time range: {time_range}")

            # --- Prediction Section ---
            st.subheader("📊 7-Day Stock Price Forecast (Open, High, Low, Close)")
            prediction_df, r2, mae, mse, rmse = enhanced_model(stock_data.copy())

            st.line_chart(prediction_df.set_index("Date")[["Open", "High", "Low", "Close"]])
            st.dataframe(prediction_df)

            st.markdown("---")
            st.subheader("📈 Model Evaluation")
            st.write(f"✅ **R² Score (Accuracy %):** {r2 * 100:.2f}%")
            st.write(f"📉 **MAE:** {mae:.2f}")
            st.write(f"📉 **MSE:** {mse:.2f}")
            st.write(f"📉 **RMSE:** {rmse:.2f}")

    else:
        page = st.sidebar.radio("Select Page", ["Login", "Register"])
        if page == "Register":
            register_user()
        else:
            login_user()


if __name__ == "__main__":
    main()