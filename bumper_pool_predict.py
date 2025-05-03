# --------------------------------------------------------------------------
# bumper_pool_predict.py — Live Google Sheet + Full Betting Logic
# --------------------------------------------------------------------------
import argparse
from datetime import datetime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# 🔗 STEP 1: Replace this URL with your Google Sheet (shareable CSV version) SHARED CSV
GOOGLE_SHEET_CSV_URL = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRWwh0ivmbEFbGOR3EsIAwWnPhXL9e5Ua6f98WJdkkkNS-Q_BHeIRUM56Y_OtC0DRGrdgAGODmbswnu/pub?gid=115312881&single=true&output=csv"

# ---------------- Helpers ----------------
def opposite_side(side: str) -> str:
    return "TV Side" if side == "Window Side" else "Window Side"

def prob_to_american(p: float) -> str:
    if p <= 0 or p >= 1:
        return "∞"
    return -round((p / (1 - p)) * 100) if p > 0.5 else +round((1 / p - 1) * 100)

def fmt_odds(o):
    return f"{o:+}" if isinstance(o, (int, float)) else str(o)

def apply_vig(p1, p2, total_prob=1.05):
    scale = total_prob / (p1 + p2)
    return p1 * scale, p2 * scale

def load_full(url: str) -> pd.DataFrame:
    raw = pd.read_csv(url)
    WIN = "Winner First Name (Use actual names for consistency in data collection)"
    LOS = "Loser First Name  (Use actual names for consistency in data collection)"
    BREAK = "Which side of the table did the winner break from?"
    MARGIN = "Balls left on table by loser"

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"])
    raw["hour_of_day"] = raw["Timestamp"].dt.hour
    raw["day_of_week"] = raw["Timestamp"].dt.day_name()

    rows = []
    for _, r in raw.iterrows():
        winner, loser = r[WIN], r[LOS]
        w_side = r[BREAK]
        l_side = opposite_side(w_side)
        margin = r[MARGIN]

        base = {
            "hour_of_day": r["hour_of_day"],
            "day_of_week": r["day_of_week"],
            "inebriated":  r["Players Inebriated?"],
        }

        rows.append({**base, "playerA": winner, "playerB": loser,
                     "break_sideA": w_side, "break_sideB": l_side,
                     "margin": margin, "y": 1})
        rows.append({**base, "playerA": loser, "playerB": winner,
                     "break_sideA": l_side, "break_sideB": w_side,
                     "margin": -margin, "y": 0})
    return pd.DataFrame(rows)

def build_model(df: pd.DataFrame) -> Pipeline:
    X = df.drop(columns=["y", "margin"])
    y = df["y"]
    cat_cols = ["playerA", "playerB", "break_sideA", "break_sideB", "inebriated", "day_of_week"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")
    return Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)

# ---------------- CLI ----------------
def parse_args():
    p = argparse.ArgumentParser(description="🎱 Predict bumper pool match odds")
    p.add_argument("--playerA", default="Austin")
    p.add_argument("--playerB", default="Brett")
    p.add_argument("--break_side", default="Window Side", choices=["Window Side", "TV Side"])
    p.add_argument("--inebriated", default="Yes", choices=["Yes", "No"])
    p.add_argument("--vig", action="store_true", help="Add sportsbook-style vig to odds")
    return p.parse_args()

# ---------------- Main ----------------
if __name__ == "__main__":
    args = parse_args()
    now = datetime.now()
    breakA = args.break_side
    breakB = opposite_side(args.break_side)

    df = load_full(GOOGLE_SHEET_CSV_URL)
    model = build_model(df)

    row = pd.DataFrame({
        "playerA":     [args.playerA],
        "playerB":     [args.playerB],
        "break_sideA": [breakA],
        "break_sideB": [breakB],
        "inebriated":  [args.inebriated],
        "hour_of_day": [now.hour],
        "day_of_week": [now.strftime("%A")],
    })

    pA = model.predict_proba(row)[0, 1]
    pB = 1 - pA

    if args.vig:
        pA_vig, pB_vig = apply_vig(pA, pB, total_prob=1.05)
        mlA, mlB = prob_to_american(pA_vig), prob_to_american(pB_vig)
    else:
        mlA, mlB = prob_to_american(pA), prob_to_american(pB)

    # Spread logic
    matchup_df = df[(df["playerA"] == args.playerA) & (df["playerB"] == args.playerB)]
    fair_spread = (
        matchup_df["margin"].mean()
        if not matchup_df.empty
        else df[df["playerA"] == args.playerA]["margin"].mean()
    )
    fair_spread = round(fair_spread, 1)

    coverA = (
        (matchup_df["margin"] > fair_spread).mean()
        if not matchup_df.empty
        else (df[df["playerA"] == args.playerA]["margin"] > fair_spread).mean()
    )
    coverB = 1 - coverA

    if args.vig:
        coverA_vig, coverB_vig = apply_vig(coverA, coverB, total_prob=1.05)
        spA, spB = prob_to_american(coverA_vig), prob_to_american(coverB_vig)
    else:
        spA, spB = prob_to_american(coverA), prob_to_american(coverB)

    # Sweep odds
    sweepA = (df[df["playerA"] == args.playerA]["margin"] == 5).mean()
    sweepB = (df[df["playerA"] == args.playerB]["margin"] == 5).mean()
    swA, swB = prob_to_american(sweepA), prob_to_american(sweepB)

    # Output
    print(f"\n🎱 **BUMPER POOL LIVE ODDS**")
    print(f"📊 Matchup: **{args.playerA}** vs **{args.playerB}**\n")

    print("💰 **Moneyline Odds**")
    print(f"   {args.playerA}: {fmt_odds(mlA)}  (p = {pA*100:.1f}%)")
    print(f"   {args.playerB}: {fmt_odds(mlB)}  (p = {pB*100:.1f}%)\n")

    print("📏 **Spread Odds**")
    print(f"   Fair Spread: {args.playerA} {fair_spread:+.1f}")
    print(f"   {args.playerA} {fair_spread:+.1f}: {fmt_odds(spA)}  (p = {coverA*100:.1f}%)")
    print(f"   {args.playerB} {-fair_spread:+.1f}: {fmt_odds(spB)}  (p = {coverB*100:.1f}%)\n")

    print("🧹 **Sweep Odds (win 5‑0)**")
    print(f"   {args.playerA}: {fmt_odds(swA)}  (p = {sweepA*100:.1f}%)")
    print(f"   {args.playerB}: {fmt_odds(swB)}  (p = {sweepB*100:.1f}%)\n")
