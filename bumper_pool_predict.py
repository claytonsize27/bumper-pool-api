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

    required_columns = [
        "Winner First Name (Use actual names for consistency in data collection)",
        "Loser First Name  (Use actual names for consistency in data collection)",
        "Which side of the table did the winner break from?",
        "Balls left on table by loser",
        "Players Inebriated?",
        "Timestamp"
    ]
    for col in required_columns:
        if col not in raw.columns:
            raise ValueError(f"Missing column: {col}")

    WIN = required_columns[0]
    LOS = required_columns[1]
    BREAK = required_columns[2]
    MARGIN = required_columns[3]

    raw["Timestamp"] = pd.to_datetime(raw["Timestamp"], errors="coerce")
    raw = raw.dropna(subset=["Timestamp", WIN, LOS, BREAK, MARGIN])

    raw["hour_of_day"] = raw["Timestamp"].dt.hour
    raw["day_of_week"] = raw["Timestamp"].dt.day_name()

    rows = []
    for _, r in raw.iterrows():
        try:
            winner, loser = r[WIN], r[LOS]
            w_side = r[BREAK]
            l_side = opposite_side(w_side)
            margin = int(r[MARGIN])
            inebriated = r["Players Inebriated?"]

            base = {
                "hour_of_day": r["hour_of_day"],
                "day_of_week": r["day_of_week"],
                "inebriated": inebriated,
            }

            rows.append({**base, "playerA": winner, "playerB": loser,
                         "break_sideA": w_side, "break_sideB": l_side,
                         "margin": margin, "y": 1})
            rows.append({**base, "playerA": loser, "playerB": winner,
                         "break_sideA": l_side, "break_sideB": w_side,
                         "margin": -margin, "y": 0})
        except Exception as e:
            print(f"Skipping row due to error: {e}")

    return pd.DataFrame(rows)

def build_models(df: pd.DataFrame):
    if df.empty:
        raise ValueError("Data is empty")

    X = df.drop(columns=["y", "margin"])
    y = df["y"]

    cat_cols = ["playerA", "playerB", "break_sideA", "break_sideB", "inebriated", "day_of_week"]
    pre = ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)],
                            remainder="passthrough")

    clf = Pipeline([("pre", pre), ("clf", LogisticRegression(max_iter=500))]).fit(X, y)
    reg = Pipeline([("pre", pre), ("reg", LogisticRegression(max_iter=500))]).fit(X, df["margin"])
    return clf, reg
