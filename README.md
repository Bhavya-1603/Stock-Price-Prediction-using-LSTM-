````markdown
# 📈 Stock Price Prediction Using Deep Learning

This project is a Flask-based web application that predicts stock price trends using a Deep Learning LSTM (Long Short-Term Memory) model.

The application fetches real-time historical stock market data using Yahoo Finance, analyzes the data, and visualizes stock price predictions with graphs.

---

## 🚀 Features

- Predict stock price trends using LSTM
- Fetch real-time stock data using Yahoo Finance
- Visualize:
  - Closing Price Trends
  - EMA 20 & EMA 50
  - EMA 100 & EMA 200
  - Predicted vs Actual Prices
- Download stock dataset as CSV
- Simple and interactive Flask web interface

---

## 🛠️ Technologies Used

- Python
- Flask
- TensorFlow / Keras
- Pandas & NumPy
- Matplotlib
- Scikit-learn
- yfinance

---

## 📂 Project Structure

```text
Stock_Price_Prediction
│
├── Website
│   ├── app.py
│   ├── stock_dl_model.h5
│   ├── templates
│   └── static
│
├── Dataset
├── Models
├── README.md
└── requirements.txt
````

---

## ⚙️ Installation

Clone the repository:

```bash
git clone <your-github-repository-link>
cd Stock_Price_Prediction-main
```

Create virtual environment:

```bash
python -m venv .venv
```

Activate environment:

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Application

Move into Website folder:

```bash
cd Website
```

Run Flask app:

```bash
python app.py
```

---

## 🌐 Open in Browser

```text
http://127.0.0.1:5000
```

---

## 📌 Example Stock Symbols

* AAPL
* GOOG
* TSLA
* MSFT
* AMZN

---

## 📊 Output

The application generates:

* EMA charts
* Prediction graphs
* Downloadable CSV dataset

---

## 👨‍💻 Author

Developed for learning and educational purposes using Deep Learning and Flask.

```
```
