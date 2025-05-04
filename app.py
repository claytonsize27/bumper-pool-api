from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
import os
import pandas as pd
from datetime import datetime
from scipy.stats import norm

from bumper_pool_predict import (
    load_full, build_models, opposite_side,
    apply_vig, prob_to_american, fmt_odds
)

# Create app FIRST
app = FastAPI()

# ✅ CORSMiddleware must be added immediately after app creation
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://claytonsize27.github.io"],  # must be exact
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Load data and models
CSV_URL = os.getenv("CSV_URL") or "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWwh0ivmbEFbGOR3EsIAwWnPhXL9e5Ua6f98WJdkkkNS-Q_BHeIRUM56Y_OtC0DRGrdgAGODmbswnu/pub?gid=115312881&single=true&output=csv"
df = load_full(CSV_URL)
model_clf, model_reg = build_models(df)

@app.get("/")
def home():
    return {"status": "OK", "message": "Bumper Pool API is live 🎱"}

@app.get("/predict")
def predict(
    playerA: str = Query(...),
    playerB: str = Query(...),
    break_side: str = Query(..., enum=["Window Side", "TV Side"]),
    inebriated: str = Query("Yes", enum=["Yes", "No"]),
    vig: bool = False
):
    now = datetime.now()
    breakB = opposite_side(break_side)
    row = pd.DataFrame({
        "playerA":     [playerA],
        "playerB":     [playerB],
        "break_sideA": [break_side],
        "break_sideB": [breakB],
        "inebriated":  [inebriated],
        "hour_of_day": [now.hour],
        "day_of_week": [now.strftime("%A")],
    })

    pA = model_clf.predict_proba(row)[0, 1]
    pB = 1 - pA
    margin_pred = model_reg.predict(row)[0]

    if vig:
        pA, pB = apply_vig(pA, pB)

    mlA, mlB = prob_to_american(pA), prob_to_american(pB)

    std_margin = df["margin"].std()
    sweepA_prob = 1 - norm.cdf(5, loc=margin_pred, scale=std_margin)
    sweepB_prob = 1 - norm.cdf(5, loc=-margin_pred, scale=std_margin)
    sweepA_odds = fmt_odds(prob_to_american(sweepA_prob))
    sweepB_odds = fmt_odds(prob_to_american(sweepB_prob))

    return {
        "moneyline": {
            playerA: mlA,
            playerB: mlB,
            "probabilities": {
                playerA: round(pA, 4),
                playerB: round(pB, 4)
            }
        },
        "sweep_odds": {
            playerA: sweepA_odds,
            playerB: sweepB_odds
        },
        "predicted_margin": {
            "winner": playerA if pA > pB else playerB,
            "loser": playerB if pA > pB else playerA,
            "margin": round(abs(margin_pred), 2)
        }
    }
