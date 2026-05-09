import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.layers import LSTM
from flask import Flask, render_template, request, send_file
import datetime as dt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import os

# -----------------------------
# Matplotlib Style
# -----------------------------
plt.style.use("ggplot")

# -----------------------------
# Flask App
# -----------------------------
app = Flask(__name__)

# -----------------------------
# Load Model
# -----------------------------
custom_objects = {
    'LSTM': lambda **kwargs: LSTM(
        **{k: v for k, v in kwargs.items() if k != 'time_major'}
    )
}

model = load_model(
    'stock_dl_model.h5',
    custom_objects=custom_objects
)

# -----------------------------
# Home Route
# -----------------------------
@app.route('/', methods=['GET', 'POST'])
def index():

    if request.method == 'POST':

        # Get Stock Symbol
        stock = request.form.get('stock')

        # Default Stock
        if not stock:
            stock = 'AAPL'

        stock = stock.upper()

        # -----------------------------
        # Date Range
        # -----------------------------
        start = dt.datetime(2000, 1, 1)
        end = dt.datetime.now()

        # -----------------------------
        # Download Stock Data
        # -----------------------------
        df = yf.download(stock, start=start, end=end)

        # Check if stock exists
        if df.empty:
            return render_template(
                'index.html',
                error="Invalid stock symbol or no data found."
            )

        # -----------------------------
        # Descriptive Statistics
        # -----------------------------
        data_desc = df.describe()

        # -----------------------------
        # Exponential Moving Averages
        # -----------------------------
        ema20 = df['Close'].ewm(span=20, adjust=False).mean()
        ema50 = df['Close'].ewm(span=50, adjust=False).mean()

        ema100 = df['Close'].ewm(span=100, adjust=False).mean()
        ema200 = df['Close'].ewm(span=200, adjust=False).mean()

        # -----------------------------
        # Train/Test Split
        # -----------------------------
        data_training = pd.DataFrame(
            df['Close'][0:int(len(df) * 0.70)]
        )

        data_testing = pd.DataFrame(
            df['Close'][int(len(df) * 0.70):]
        )

        # -----------------------------
        # Scaling Data
        # -----------------------------
        scaler = MinMaxScaler(feature_range=(0, 1))

        data_training_array = scaler.fit_transform(data_training)

        # -----------------------------
        # Prepare Testing Data
        # -----------------------------
        past_100_days = data_training.tail(100)

        final_df = pd.concat(
            [past_100_days, data_testing],
            ignore_index=True
        )

        input_data = scaler.fit_transform(final_df)

        x_test = []
        y_test = []

        for i in range(100, input_data.shape[0]):

            x_test.append(input_data[i - 100:i])

            y_test.append(input_data[i, 0])

        x_test = np.array(x_test)
        y_test = np.array(y_test)

        # -----------------------------
        # Prediction
        # -----------------------------
        y_predicted = model.predict(x_test)

        # -----------------------------
        # Reverse Scaling
        # -----------------------------
        scale_factor = 1 / scaler.scale_[0]

        y_predicted = y_predicted * scale_factor
        y_test = y_test * scale_factor

        # Create Static Folder if Missing
        os.makedirs("static", exist_ok=True)

        # -----------------------------
        # Plot 1 : EMA 20 & 50
        # -----------------------------
        fig1, ax1 = plt.subplots(figsize=(12, 6))

        ax1.plot(df['Close'], label='Closing Price')
        ax1.plot(ema20, label='EMA 20')
        ax1.plot(ema50, label='EMA 50')

        ax1.set_title(f"{stock} Closing Price (EMA 20 & EMA 50)")
        ax1.set_xlabel("Date")
        ax1.set_ylabel("Price")
        ax1.legend()

        ema_chart_20_50 = "static/ema_20_50.png"

        fig1.savefig(ema_chart_20_50)

        plt.close(fig1)

        # -----------------------------
        # Plot 2 : EMA 100 & 200
        # -----------------------------
        fig2, ax2 = plt.subplots(figsize=(12, 6))

        ax2.plot(df['Close'], label='Closing Price')
        ax2.plot(ema100, label='EMA 100')
        ax2.plot(ema200, label='EMA 200')

        ax2.set_title(f"{stock} Closing Price (EMA 100 & EMA 200)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()

        ema_chart_100_200 = "static/ema_100_200.png"

        fig2.savefig(ema_chart_100_200)

        plt.close(fig2)

        # -----------------------------
        # Plot 3 : Prediction Graph
        # -----------------------------
        fig3, ax3 = plt.subplots(figsize=(12, 6))

        ax3.plot(
            y_test,
            label='Original Price',
            linewidth=1
        )

        ax3.plot(
            y_predicted,
            label='Predicted Price',
            linewidth=1
        )

        ax3.set_title(f"{stock} Prediction vs Original")
        ax3.set_xlabel("Time")
        ax3.set_ylabel("Price")
        ax3.legend()

        prediction_chart = "static/stock_prediction.png"

        fig3.savefig(prediction_chart)

        plt.close(fig3)

        # -----------------------------
        # Save CSV Dataset
        # -----------------------------
        csv_file_path = f"static/{stock}_dataset.csv"

        df.to_csv(csv_file_path)

        # -----------------------------
        # Render Template
        # -----------------------------
        return render_template(
            'index.html',

            stock=stock,

            plot_path_ema_20_50=ema_chart_20_50,

            plot_path_ema_100_200=ema_chart_100_200,

            plot_path_prediction=prediction_chart,

            data_desc=data_desc.to_html(
                classes='table table-bordered'
            ),

            dataset_link=f"/download/{stock}_dataset.csv"
        )

    # GET Request
    return render_template('index.html')


# -----------------------------
# Download Route
# -----------------------------
@app.route('/download/<filename>')
def download_file(filename):

    return send_file(
        f"static/{filename}",
        as_attachment=True
    )


# -----------------------------
# Run App
# -----------------------------
if __name__ == '__main__':

    app.run(
        debug=True,
        host='0.0.0.0',
        port=5000
    )