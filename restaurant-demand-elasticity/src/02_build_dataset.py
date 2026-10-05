"""
Step 2 - Clean the FRED file and build the monthly analysis dataset.

Output: data/processed/restaurant_monthly.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "fredgraph.csv"
OUT = ROOT / "data" / "processed" / "restaurant_monthly.csv"

# Restaurant dining rooms closed or restricted: flagged and excluded later.
COVID_START, COVID_END = "2020-03-01", "2021-06-01"


def main() -> None:
    df = pd.read_csv(RAW, na_values=[".", ""])
    date_col = "observation_date" if "observation_date" in df.columns else df.columns[0]
    df = df.rename(
        columns={
            date_col: "date",
            "RSFSDP": "sales_nominal",
            "CUSR0000SEFV": "cpi_menu",
            "CUSR0000SAF11": "cpi_grocery",
            "CPIAUCSL": "cpi_all",
            "DSPIC96": "income_real",
            "CES7000000008": "wage_nominal",
            "WPU02": "ppi_food",
        }
    )
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    # October 2025 CPI values are missing from the source series; fill single
    # one-month gaps by linear interpolation and flag them.
    for c in ["cpi_menu", "cpi_grocery", "cpi_all"]:
        df[f"{c}_interpolated"] = df[c].isna() & df[c].shift(1).notna() & df[c].shift(-1).notna()
        df[c] = df[c].interpolate(limit=1, limit_area="inside")
    df["cpi_interpolated"] = df[[f"{c}_interpolated" for c in ["cpi_menu", "cpi_grocery", "cpi_all"]]].any(axis=1)
    df = df.drop(columns=[f"{c}_interpolated" for c in ["cpi_menu", "cpi_grocery", "cpi_all"]])

    need = ["sales_nominal", "cpi_menu", "cpi_grocery", "cpi_all", "income_real", "wage_nominal", "ppi_food"]
    df = df.dropna(subset=need).reset_index(drop=True)

    # Base: latest complete calendar year
    full = df.groupby(df["date"].dt.year)["date"].count()
    base_year = int(full[full == 12].index.max())
    base = df[df["date"].dt.year == base_year]
    df["base_year"] = base_year

    # Quantity = real restaurant spending (sales deflated by menu prices),
    # expressed in base-year dollars.
    df["sales_real"] = df["sales_nominal"] / df["cpi_menu"] * base["cpi_menu"].mean()
    # Prices relative to all other goods (real prices), indexed to base year = 100
    rel = lambda s: s / df["cpi_all"]
    for name, col in [("p_menu", "cpi_menu"), ("p_grocery", "cpi_grocery"), ("ppi_real", "ppi_food")]:
        r = rel(df[col])
        df[name] = 100 * r / r[df["date"].dt.year == base_year].mean()
    df["wage_real"] = df["wage_nominal"] * base["cpi_all"].mean() / df["cpi_all"]

    df["ln_q"] = np.log(df["sales_real"])
    df["ln_p"] = np.log(df["p_menu"])
    df["ln_pg"] = np.log(df["p_grocery"])
    df["ln_y"] = np.log(df["income_real"])
    df["ln_w"] = np.log(df["wage_real"])
    df["ln_ppi"] = np.log(df["ppi_real"])
    df["trend"] = np.arange(len(df)) / 12.0
    df["covid"] = df["date"].between(COVID_START, COVID_END).astype(int)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False, float_format="%.6f")
    print(
        f"wrote {OUT.relative_to(ROOT)}: {len(df)} months, "
        f"{df['date'].min():%Y-%m} to {df['date'].max():%Y-%m}, base year {base_year}"
    )


if __name__ == "__main__":
    main()
