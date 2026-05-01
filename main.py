import yfinance as yf
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI(title="Gold Price Prediction API")
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

class GoldInput(BaseModel):
    SPX: float
    USO: float
    SLV: float
    EUR_USD: float

@app.get("/")
def home():
    return {"message": "Gold Price Prediction API is running"}

@app.post("/predict")
def predict(data: GoldInput):
    features = np.array([[data.SPX, data.USO, data.SLV, data.EUR_USD]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)
    return {"predicted_GLD_price": float(prediction[0])}
def fetch_live_data():
    spx = yf.Ticker("^GSPC").history(period="1d")["Close"].iloc[-1]
    uso = yf.Ticker("USO").history(period="1d")["Close"].iloc[-1]
    slv = yf.Ticker("SLV").history(period="1d")["Close"].iloc[-1]
    eur_usd = yf.Ticker("EURUSD=X").history(period="1d")["Close"].iloc[-1]
    usd_inr = yf.Ticker("INR=X").history(period="1d")["Close"].iloc[-1]
    return spx, uso, slv, eur_usd, usd_inr
@app.get("/predict-live")
def predict_live():
    spx, uso, slv, eur_usd, usd_inr = fetch_live_data()

    features = np.array([[spx, uso, slv, eur_usd]])
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)

    predicted_usd = float(prediction[0])
    predicted_inr = predicted_usd * float(usd_inr)

    return {
        "live_data": {
            "SPX": float(spx),
            "USO": float(uso),
            "SLV": float(slv),
            "EUR_USD": float(eur_usd),
            "USD_INR": float(usd_inr)
        },
        "predicted_GLD_price_usd": predicted_usd,
        "predicted_GLD_price_inr": predicted_inr
    }