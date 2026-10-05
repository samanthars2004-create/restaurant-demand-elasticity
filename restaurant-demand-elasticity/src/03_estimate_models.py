"""
Step 3 - Estimate U.S. restaurant demand (own-price elasticity of eating out,
cross-price elasticity with groceries), validate it, and produce every table
and figure used in the paper.

Preferred model: 12-month log differences, two-stage least squares.
    d ln q = a + b d ln p_menu + c d ln p_grocery + g d ln income + e
where d is the change versus the same month a year earlier, and the menu
price is instrumented with restaurant cost shifters (real leisure &
hospitality wages, real processed-food producer prices).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from econometrics import iv2sls, long_run, ols  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed" / "restaurant_monthly.csv"
TAB = ROOT / "output" / "tables"
FIG = ROOT / "output" / "figures"
INSTR = ["d_ln_w", "d_ln_ppi"]

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size": 10,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "lines.linewidth": 1.6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


# ----------------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------------
def load() -> pd.DataFrame:
    df = pd.read_csv(DATA, parse_dates=["date"])
    for v in ["ln_q", "ln_p", "ln_pg", "ln_y", "ln_w", "ln_ppi"]:
        df[f"d_{v}"] = df[v] - df[v].shift(12)
    # A 12-month change is unusable if either endpoint is a COVID month.
    df["covid_diff"] = (df["covid"] == 1) | (df["covid"].shift(12) == 1)
    df["covid_level"] = (df["covid"] == 1) | (df["covid"].shift(1) == 1)
    df["ln_q_lag"] = df["ln_q"].shift(1)
    return df


def diff_sample(df: pd.DataFrame, start="1900-01-01", end="2100-01-01") -> pd.DataFrame:
    s = df[(df["date"] >= start) & (df["date"] <= end) & ~df["covid_diff"]]
    return s.dropna(subset=["d_ln_q", "d_ln_p", "d_ln_pg", "d_ln_y"] + INSTR)


# ----------------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------------
def fit_diff(s: pd.DataFrame, iv: bool = True, instruments=INSTR, name=""):
    exog = pd.DataFrame({"d_ln_pg": s["d_ln_pg"], "d_ln_y": s["d_ln_y"], "const": 1.0}, index=s.index)
    if iv:
        return iv2sls(s["d_ln_q"], exog, s[["d_ln_p"]], s[instruments], name=name)
    return ols(s["d_ln_q"], pd.concat([s[["d_ln_p"]], exog], axis=1), name=name)


def summarize(r, p="d_ln_p", pg="d_ln_pg", y="d_ln_y") -> dict:
    return {
        "own": r.params[p], "own_se": r.bse[p],
        "cross": r.params[pg], "cross_se": r.bse[pg],
        "income": r.params[y], "income_se": r.bse[y],
        "n": r.nobs, "r2": r.r2, "F": r.first_stage_F,
    }


def stars(c, se):
    t = abs(c / se) if se else 0
    return "***" if t > 2.576 else "**" if t > 1.96 else "*" if t > 1.645 else ""


def fmt(c, se):
    if c is None or (isinstance(c, float) and np.isnan(c)):
        return ""
    return f"{c:.3f}{stars(c, se)} ({se:.3f})"


def to_markdown(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join([str(df.index.name or "")] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [str(v) for v in row]) + " |")
    return "\n".join(lines) + "\n"


def save_table(df: pd.DataFrame, name: str) -> None:
    df.to_csv(TAB / f"{name}.csv")
    (TAB / f"{name}.md").write_text(to_markdown(df))


def table_from(rows: list[dict], index_label: str) -> pd.DataFrame:
    raw = pd.DataFrame(rows)
    t = pd.DataFrame(
        {
            "Menu-price elasticity": [fmt(a, b) for a, b in zip(raw.own, raw.own_se)],
            "Grocery-price (cross) elasticity": [fmt(a, b) for a, b in zip(raw.cross, raw.cross_se)],
            "Income elasticity": [fmt(a, b) for a, b in zip(raw.income, raw.income_se)],
            "First-stage F": ["" if pd.isna(f) else f"{f:.1f}" for f in raw.F],
            "Months": raw.n.values,
        },
        index=raw[index_label].values,
    )
    t.index.name = index_label
    return t


# ----------------------------------------------------------------------------
def descriptive(s: pd.DataFrame) -> pd.DataFrame:
    by = int(s["base_year"].iloc[0])
    cols = {
        "sales_real": f"Real restaurant spending ($ millions/month, {by} $)",
        "sales_nominal": "Nominal restaurant sales ($ millions/month)",
        "p_menu": f"Real menu price index ({by} = 100)",
        "p_grocery": f"Real grocery price index ({by} = 100)",
        "income_real": "Real disposable income (bil. chained 2017 $)",
        "wage_real": f"Real leisure & hospitality wage ($/hour, {by} $)",
        "d_ln_q": "12-month change in real restaurant spending (log points)",
        "d_ln_p": "12-month change in real menu price (log points)",
    }
    d = s[list(cols)].describe().T[["mean", "std", "min", "max"]]
    d.index = [cols[c] for c in d.index]
    d.index.name = "Variable"
    d = d.apply(lambda r: r.round(3) if "change" in r.name else r.round(1), axis=1)
    save_table(d, "table1_descriptive")
    return d


def main_results(s: pd.DataFrame):
    specs = [
        ("(1) OLS", False, INSTR),
        ("(2) IV: wages + food PPI", True, INSTR),
        ("(3) IV: wages only", True, ["d_ln_w"]),
        ("(4) IV: food PPI only", True, ["d_ln_ppi"]),
    ]
    rows, fits = [], {}
    for name, iv, ins in specs:
        r = fit_diff(s, iv, ins, name=name)
        fits[name] = r
        rows.append({"Model": name, **summarize(r)})
    raw = pd.DataFrame(rows).set_index("Model")
    raw.to_csv(TAB / "table2_main_results_raw.csv")
    t = table_from(rows, "Model").T
    t.index.name = ""
    save_table(t, "table2_main_results")
    return raw, fits


def subperiods(df: pd.DataFrame) -> pd.DataFrame:
    last = f"{df['date'].max():%Y}"
    periods = [
        ("1993-2007", "1993-01-01", "2007-12-01"),
        ("2008-2019", "2008-01-01", "2019-12-01"),
        ("1993-2019 (pre-COVID)", "1993-01-01", "2019-12-01"),
        (f"2022-{last} (post-COVID)", "2022-01-01", "2100-01-01"),
        (f"Full sample 1993-{last}", "1900-01-01", "2100-01-01"),
    ]
    rows = []
    for label, a, b in periods:
        r = fit_diff(diff_sample(df, a, b), True)
        rows.append({"Sample": label, **summarize(r)})
    raw = pd.DataFrame(rows)
    raw.to_csv(TAB / "table3_subperiods_raw.csv", index=False)
    save_table(table_from(rows, "Sample"), "table3_subperiods")
    return raw


def alt_specs(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lv = df[~df["covid_level"]].dropna(subset=["ln_q_lag"])

    def levels(trend: bool, dynamic: bool, iv: bool, label: str):
        exog = pd.DataFrame({"ln_pg": lv["ln_pg"], "ln_y": lv["ln_y"], "const": 1.0}, index=lv.index)
        if trend:
            exog["trend"] = lv["trend"]
        if dynamic:
            exog["lag"] = lv["ln_q_lag"]
        if iv:
            r = iv2sls(lv["ln_q"], exog, lv[["ln_p"]], lv[["ln_w", "ln_ppi"]], name=label)
        else:
            r = ols(lv["ln_q"], pd.concat([lv[["ln_p"]], exog], axis=1), name=label)
        out = {"Specification": label, **summarize(r, "ln_p", "ln_pg", "ln_y")}
        if dynamic:
            out["lr"], out["lr_se"] = long_run(r, "ln_p", "lag")
        return out

    rows.append(levels(True, False, False, "Levels with trend, OLS"))
    rows.append(levels(True, False, True, "Levels with trend, IV"))
    rows.append(levels(False, False, True, "Levels without trend, IV"))
    rows.append(levels(True, True, True, "Levels with trend, dynamic IV (long run)"))

    # 6-month differences (shorter horizon); same COVID endpoint rule
    d6 = df.copy()
    for v in ["ln_q", "ln_p", "ln_pg", "ln_y", "ln_w", "ln_ppi"]:
        d6[f"d_{v}"] = d6[v] - d6[v].shift(6)
    d6 = d6[~((d6["covid"] == 1) | (d6["covid"].shift(6) == 1))].dropna(subset=["d_ln_q", "d_ln_p"] + INSTR)
    r = fit_diff(d6, True, name="6m")
    rows.append({"Specification": "6-month differences, IV", **summarize(r)})

    raw = pd.DataFrame(rows)
    # For the dynamic model, report the long-run price effect in the price column
    dyn = raw["Specification"].str.contains("dynamic")
    raw.loc[dyn, "own"], raw.loc[dyn, "own_se"] = raw.loc[dyn, "lr"], raw.loc[dyn, "lr_se"]
    raw.to_csv(TAB / "table4_alt_specs_raw.csv", index=False)
    save_table(table_from(raw.to_dict("records"), "Specification"), "table4_alt_specs")
    return raw


def out_of_sample(df: pd.DataFrame):
    """
    Train on 1993-2019, then predict each month's 12-month change in real
    restaurant spending from July 2022 (first change with both endpoints
    after the excluded COVID months) using actual price and income changes.
    """
    train = diff_sample(df, "1993-01-01", "2019-12-01")
    test = diff_sample(df, "2022-07-01", "2100-01-01").copy()
    preds = {}
    r = fit_diff(train, True, name="pref")
    X = pd.DataFrame({"d_ln_p": test.d_ln_p, "d_ln_pg": test.d_ln_pg, "d_ln_y": test.d_ln_y, "const": 1.0})
    preds["Price model (IV), trained 1993-2019"] = X[r.params.index].to_numpy() @ r.params.to_numpy()
    ri = ols(train["d_ln_q"], pd.DataFrame({"d_ln_y": train.d_ln_y, "const": 1.0}), name="inc")
    preds["Income-only model, trained 1993-2019"] = (
        pd.DataFrame({"d_ln_y": test.d_ln_y, "const": 1.0}).to_numpy() @ ri.params.to_numpy()
    )
    preds["Average growth 1993-2019"] = np.full(len(test), train["d_ln_q"].mean())
    actual = test["d_ln_q"].to_numpy()
    rows = []
    for name, p in preds.items():
        err = (actual - p) * 100
        rows.append(
            {
                "Model": name,
                "RMSE (pct. points)": round(float(np.sqrt(np.mean(err**2))), 2),
                "Mean abs. error": round(float(np.mean(np.abs(err))), 2),
                "Mean error (bias)": round(float(np.mean(err)), 2),
                "Months": len(test),
            }
        )
    t = pd.DataFrame(rows).set_index("Model")
    save_table(t, "table5_out_of_sample")
    path = pd.DataFrame({"date": test["date"].values, "actual": actual * 100})
    for k, v in preds.items():
        path[k] = v * 100
    path.to_csv(TAB / "out_of_sample_forecasts.csv", index=False)
    return t, path


def rolling(df: pd.DataFrame, years: int = 10) -> pd.DataFrame:
    s = diff_sample(df).reset_index(drop=True)
    w = years * 12
    rows = []
    for end in range(w, len(s) + 1, 6):
        sub = s.iloc[end - w : end]
        span = (sub["date"].iloc[-1] - sub["date"].iloc[0]).days / 365.25
        if span > years + 2:
            continue
        r = fit_diff(sub, True)
        rows.append(
            {
                "window_end": sub["date"].iloc[-1],
                "elasticity": r.params["d_ln_p"],
                "lo": r.params["d_ln_p"] - 1.96 * r.bse["d_ln_p"],
                "hi": r.params["d_ln_p"] + 1.96 * r.bse["d_ln_p"],
                "F": r.first_stage_F,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(TAB / "rolling_elasticity.csv", index=False)
    return out


def scenarios(df: pd.DataFrame, own: float, cross: float, pass_w: float) -> pd.DataFrame:
    last12 = df.tail(12)
    rev0 = last12["sales_nominal"].sum() / 1000  # $ billions per year
    rows = []
    cases = [
        ("Industry-wide menu prices +5%", 0.05, 0.0),
        ("Industry-wide menu prices +10%", 0.10, 0.0),
        ("Real restaurant wages +10% (passed into menu prices)", (1.10) ** pass_w - 1, 0.0),
        ("Grocery prices +10%", 0.0, 0.10),
    ]
    for label, dp, dpg in cases:
        dq = (1 + dp) ** own * (1 + dpg) ** cross - 1
        drev = (1 + dp) * (1 + dq) - 1
        rows.append(
            {
                "Scenario": label,
                "Menu price change (%)": round(100 * dp, 1),
                "Real spending (traffic) change (%)": round(100 * dq, 1),
                "Restaurant sales change (%)": round(100 * drev, 1),
                "Sales change ($ bn/yr)": round(rev0 * drev, 1),
            }
        )
    t = pd.DataFrame(rows).set_index("Scenario")
    save_table(t, "table6_scenarios")
    t.attrs["rev0"] = rev0
    return t


# ----------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------
def fig_series(df: pd.DataFrame) -> None:
    by = int(df["base_year"].iloc[0])
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.6), sharex=True)
    axes[0].plot(df["date"], df["p_menu"], color=BLUE, label="Menu prices")
    axes[0].plot(df["date"], df["p_grocery"], color=ORANGE, label="Grocery prices")
    axes[0].set_ylabel(f"Index, {by} = 100")
    axes[0].set_title("Real menu and grocery prices (relative to all consumer prices)", loc="left", fontsize=10, color=INK)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")
    axes[1].plot(df["date"], df["sales_real"] / 1000, color=AQUA)
    axes[1].set_ylabel(f"\\$ billions/month ({by} \\$)")
    axes[1].set_title("Real restaurant spending", loc="left", fontsize=10, color=INK)
    for ax in axes:
        ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-06-30"), color=GRID, alpha=0.6, lw=0)
    axes[1].annotate("COVID months\n(excluded)", xy=(pd.Timestamp("2020-10-01"), 0.08),
                     xycoords=("data", "axes fraction"), ha="center", fontsize=8, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_prices_and_spending.png")
    plt.close(fig)


def fig_rolling(roll: pd.DataFrame, full: float) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.fill_between(roll["window_end"], roll["lo"], roll["hi"], color=BLUE, alpha=0.15, lw=0, label="95% confidence band")
    ax.plot(roll["window_end"], roll["elasticity"], color=BLUE, label="10-year rolling IV estimate")
    ax.axhline(full, color=INK2, lw=1, ls="--", label=f"Full-sample estimate ({full:.2f})")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("Menu-price elasticity")
    ax.set_xlabel("End of 10-year estimation window")
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(FIG / "fig2_rolling_elasticity.png")
    plt.close(fig)


def fig_forecast(path: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.2))
    ax.plot(path["date"], path["actual"], color=INK, lw=1.8, label="Actual")
    ax.plot(path["date"], path["Price model (IV), trained 1993-2019"], color=BLUE, label="Price model (trained 1993-2019)")
    ax.plot(path["date"], path["Income-only model, trained 1993-2019"], color=ORANGE, ls="--", label="Income-only model")
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_ylabel("12-month change in real spending (%)")
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    fig.savefig(FIG / "fig3_out_of_sample.png")
    plt.close(fig)


# ----------------------------------------------------------------------------
def main() -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load()
    s = diff_sample(df)

    desc = descriptive(s)
    main_raw, fits = main_results(s)
    sub = subperiods(df)
    alt = alt_specs(df)
    oos, path = out_of_sample(df)
    roll = rolling(df)

    pref = main_raw.loc["(2) IV: wages + food PPI"]
    # Wage pass-through into menu prices: first stage of the wages-only IV
    fs_w = fits["(3) IV: wages only"].extra["first_stage"]
    pass_w = float(fs_w.params["d_ln_w"])
    fs = fits["(2) IV: wages + food PPI"].extra["first_stage"]
    sc = scenarios(df, pref.own, pref.cross, pass_w)

    fig_series(df)
    fig_rolling(roll, pref.own)
    fig_forecast(path)

    a = sub[sub.Sample == "1993-2007"].iloc[0]
    b = sub[sub.Sample == "2008-2019"].iloc[0]
    key = {
        "sample": f"{s['date'].min():%Y-%m} to {s['date'].max():%Y-%m}",
        "n_months": int(len(s)),
        "base_year": int(df["base_year"].iloc[0]),
        "pass_through_wage": pass_w,
        "pass_through_wage_se": float(fs_w.bse["d_ln_w"]),
        "first_stage_wage": float(fs.params["d_ln_w"]),
        "first_stage_ppi": float(fs.params["d_ln_ppi"]),
        "z_period_diff": float((a.own - b.own) / np.sqrt(a.own_se**2 + b.own_se**2)),
        "baseline_sales_bn_yr": sc.attrs["rev0"],
        "rolling_min": float(roll.elasticity.min()),
        "rolling_max": float(roll.elasticity.max()),
        "rolling_min_end": f"{roll.loc[roll.elasticity.idxmin(), 'window_end']:%Y}",
        "rolling_max_end": f"{roll.loc[roll.elasticity.idxmax(), 'window_end']:%Y}",
        "rolling_last": float(roll.elasticity.iloc[-1]),
        "rolling_share_negative_sig": float((roll.hi < 0).mean()),
        "menu_price_growth_total": float(df.p_menu.iloc[-1] / df.p_menu.iloc[0] - 1),
        "grocery_price_growth_total": float(df.p_grocery.iloc[-1] / df.p_grocery.iloc[0] - 1),
    }
    (ROOT / "output" / "key_numbers.json").write_text(json.dumps(key, indent=2, default=float))

    pd.set_option("display.width", 200)
    print(desc, "\n")
    for f in ["table2_main_results", "table3_subperiods", "table4_alt_specs", "table5_out_of_sample", "table6_scenarios"]:
        print(pd.read_csv(TAB / f"{f}.csv", index_col=0).to_string(), "\n")
    print(json.dumps(key, indent=1))


if __name__ == "__main__":
    main()
