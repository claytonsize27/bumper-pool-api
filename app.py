from fastapi import FastAPI, Request, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from scipy.stats import norm
import pandas as pd
import os
import logging

from bumper_pool_predict import (
    load_full, build_models, opposite_side,
    apply_vig, prob_to_american, fmt_odds
)

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# ---------------- App Init ----------------
app = FastAPI()

origins = ["https://claytonsize27.github.io"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal Server Error"},
        headers={
            "Access-Control-Allow-Origin": origins[0],
            "Access-Control-Allow-Credentials": "true",
            "Access-Control-Allow-Methods": "*",
            "Access-Control-Allow-Headers": "*",
        },
    )

CSV_URL = os.getenv("CSV_URL") or "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWwh0ivmbEFbGOR3EsIAwWnPhXL9e5Ua6f98WJdkkkNS-Q_BHeIRUM56Y_OtC0DRGrdgAGODmbswnu/pub?gid=115312881&single=true&output=csv"
df = load_full(CSV_URL)
model_clf, model_reg = build_models(df)

@app.get("/")
def home():
    return {"status": "OK", "message": "Bumper Pool API is live 🎝️"}

@app.get("/predict")
def predict(
    playerA: str = Query(...),
    playerB: str = Query(...),
    break_side: str = Query(...),
    inebriated: str = Query("Yes"),
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

    pA = float(model_clf.predict_proba(row)[0, 1])
    pB = 1 - pA
    margin_pred = float(model_reg.predict(row)[0])

    winner = playerA if pA > pB else playerB
    loser = playerB if pA > pB else playerA
    signed_margin = margin_pred if winner == playerA else -margin_pred

    if vig:
        pA, pB = apply_vig(pA, pB)

    mlA = fmt_odds(prob_to_american(pA))
    mlB = fmt_odds(prob_to_american(pB))

    std_margin = float(df["margin"].std())
    sweepA = float(1 - norm.cdf(5, loc=(margin_pred if playerA == winner else -margin_pred), scale=std_margin))
    sweepB = float(1 - norm.cdf(5, loc=(margin_pred if playerB == winner else -margin_pred), scale=std_margin))

    margin_probs = {}
    for i in range(1, 6):
        prob_winner = float(norm.cdf(i + 0.5, loc=signed_margin, scale=std_margin) - norm.cdf(i - 0.5, loc=signed_margin, scale=std_margin))
        odds_winner = fmt_odds(prob_to_american(prob_winner))

        prob_loser = float(norm.cdf(-i + 0.5, loc=signed_margin, scale=std_margin) - norm.cdf(-i - 0.5, loc=signed_margin, scale=std_margin))
        odds_loser = fmt_odds(prob_to_american(prob_loser))

        margin_probs[str(i)] = {
            f"{winner}_probability": round(prob_winner, 4),
            f"{winner}_odds": odds_winner,
            f"{loser}_probability": round(prob_loser, 4),
            f"{loser}_odds": odds_loser
        }

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
            playerA: fmt_odds(prob_to_american(sweepA)),
            playerB: fmt_odds(prob_to_american(sweepB))
        },
        "predicted_margin": {
            "winner": winner,
            "loser": loser,
            "margin": round(abs(signed_margin), 2)
        },
        "margin_distribution": margin_probs
    }
