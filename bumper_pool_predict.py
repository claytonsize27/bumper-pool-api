from datetime import datetime
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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
    raw.columns = raw.columns.str.strip()

    WIN = "Winner First Name (Use actual names for consistency in data collection)"
    LOS = "Loser First Name  (Use actual names for consistency in data collection)"
    BREAK = "Which side of the table did the winner break from?"
    MARGIN = "Balls left on table by loser"
    INEBRIATED = "Players Inebriated?"

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw["hour_of_day"] = raw["Timestamp"].dt.hour
    raw["day_of_week"] = raw["Timestamp"].dt.day_name()

    rows = []
    for _, r in raw.iterrows():
        if pd.isna(r[WIN]) or pd.isna(r[LOS]) or pd.isna(r[BREAK]) or pd.isna(r[MARGIN]):
            continue
        winner, loser = r[WIN], r[LOS]
        w_side = r[BREAK]
        l_side = opposite_side(w_side)
        margin = r[MARGIN]

        base = {
            "hour_of_day": r["hour_of_day"],
            "day_of_week": r["day_of_week"],
            "inebriated":  r[INEBRIATED],
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
