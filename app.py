from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from bumper_pool_predict import (
    opposite_side, load_full, build_model,
    apply_vig, prob_to_american, fmt_odds
)
from datetime import datetime
import pandas as pd
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://claytonsize27.github.io"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

CSV_URL = os.getenv("CSV_URL") or "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWwh0ivmbEFbGOR3EsIAwWnPhXL9e5Ua6f98WJdkkkNS-Q_BHeIRUM56Y_OtC0DRGrdgAGODmbswnu/pub?gid=115312881&single=true&output=csv"
df = load_full(CSV_URL)
model = build_model(df)

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

    pA = model.predict_proba(row)[0, 1]
    pB = 1 - pA
    if vig:
        pA, pB = apply_vig(pA, pB)

    sweepA = (df[df["playerA"] == playerA]["margin"] == 5).mean()
    sweepB = (df[df["playerA"] == playerB]["margin"] == 5).mean()
    spread = df[df["playerA"] == playerA]["margin"].mean() - df[df["playerA"] == playerB]["margin"].mean()
    predicted_margin = round(spread, 2)

    return {
        "moneyline": {
            playerA: fmt_odds(prob_to_american(pA)),
            playerB: fmt_odds(prob_to_american(pB)),
            "probabilities": {
                playerA: round(pA, 4),
                playerB: round(pB, 4)
            }
        },
        "sweep_odds": {
            playerA: fmt_odds(prob_to_american(sweepA)),
            playerB: fmt_odds(prob_to_american(sweepB)),
        },
        "predicted_margin": {
            "winner": playerA if pA > pB else playerB,
            "loser": playerB if pA > pB else playerA,
            "margin": abs(predicted_margin) if pA != pB else 0.0
        }
    }
