"""Generate the two synthetic demo datasets in data/ (seeded, reproducible).

They are synthetic on purpose: no licensing questions, and they contain known
patterns (seasonality, a plan with higher churn, a ticket/NPS relationship,
planted outliers) so you can check the agent finds what is really there.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent.parent / "data"
rng = np.random.default_rng(42)


def subscriptions(n=1200) -> pd.DataFrame:
    plans = rng.choice(["Basic", "Pro", "Enterprise"], n, p=[0.55, 0.33, 0.12])
    fee = pd.Series(plans).map({"Basic": 499, "Pro": 1499, "Enterprise": 4999}).to_numpy()
    seats = np.where(plans == "Enterprise", rng.integers(10, 120, n),
                     np.where(plans == "Pro", rng.integers(2, 15, n), 1))
    signup = pd.Timestamp("2024-01-01") + pd.to_timedelta(rng.integers(0, 640, n), unit="D")
    region = rng.choice(["North", "South", "East", "West"], n, p=[0.3, 0.25, 0.25, 0.2])
    tickets = rng.poisson(np.where(plans == "Basic", 2.5, 1.5))
    nps = np.clip(np.round(8.5 - 0.6 * tickets + rng.normal(0, 1.4, n)), 0, 10).astype(int)
    churn_p = np.where(plans == "Basic", 0.30, np.where(plans == "Pro", 0.15, 0.07)) + 0.03 * (tickets > 4)
    status = np.where(rng.random(n) < churn_p, "cancelled",
                      rng.choice(["active", "paused", "trial"], n, p=[0.85, 0.07, 0.08]))
    monthly_revenue = (fee * seats * rng.uniform(0.85, 1.0, n)).round(0)
    monthly_revenue[status == "cancelled"] = 0
    # planted anomalies: a few enormous enterprise contracts
    big = rng.choice(np.where(plans == "Enterprise")[0], 4, replace=False)
    monthly_revenue[big] = monthly_revenue[big] * 6
    status[big] = "active"
    df = pd.DataFrame({
        "customer_id": [f"C{10000 + i}" for i in range(n)],
        "signup_date": signup.strftime("%Y-%m-%d"),
        "plan": plans, "region": region, "status": status, "seats": seats,
        "monthly_revenue": monthly_revenue, "support_tickets": tickets, "nps": nps,
    })
    return df.sort_values("signup_date").reset_index(drop=True)


def retail_sales(n=3000) -> pd.DataFrame:
    dates = pd.Timestamp("2025-01-01") + pd.to_timedelta(rng.integers(0, 365, n), unit="D")
    category = rng.choice(["Electronics", "Grocery", "Clothing", "Home"], n, p=[0.2, 0.4, 0.25, 0.15])
    base_price = pd.Series(category).map({"Electronics": 8000, "Grocery": 350, "Clothing": 1200, "Home": 2200})
    price = (base_price * rng.lognormal(0, 0.35, n)).round(0)
    season = 1 + 0.5 * np.isin(dates.month, [10, 11, 12])  # festive season bump
    units = rng.poisson(2 * season) + 1
    discount = np.round(rng.choice([0, 0.05, 0.1, 0.2], n, p=[0.5, 0.25, 0.15, 0.1]), 2)
    df = pd.DataFrame({
        "Order Date": dates.strftime("%d/%m/%Y"),
        "Store": rng.choice(["Delhi", "Mumbai", "Bengaluru", "Pune", "Jaipur"], n),
        "Category": category, "Units": units, "Unit Price": price, "Discount": discount,
        "Channel": rng.choice(["Online", "In-store"], n, p=[0.45, 0.55]),
    })
    df["Revenue"] = (df["Units"] * df["Unit Price"] * (1 - df["Discount"])).round(2)
    return df.sort_values("Order Date").reset_index(drop=True)


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    subs = subscriptions()
    subs.to_csv(OUT / "subscriptions.csv", index=False)
    sales = retail_sales()
    sales.to_excel(OUT / "retail_sales.xlsx", index=False)
    print(f"subscriptions.csv {subs.shape}, retail_sales.xlsx {sales.shape}")
