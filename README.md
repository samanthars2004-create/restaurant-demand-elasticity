# When Restaurants Raise Prices, How Much Less Do Customers Eat Out?

Estimating the price elasticity of U.S. restaurant demand with public monthly data (1993–2026), testing whether customers switch to groceries, validating the result four ways, and turning it into numbers an operator, investor or policymaker can use.

**Paper:** [SSRN link — add after upload] · **Author:** Samantha Spadaro, MBA, University at Albany (BFIN515)

---

## The problem

Restaurants have raised menu prices about 18% faster than overall inflation since 1992, mostly to cover rising labor and food costs. Operators, investors and policymakers all need to know: *when menu prices rise, how much do customers cut back, and do they switch to cooking at home?*

## Key findings

| | Estimate | What it means |
|---|---|---|
| Menu-price elasticity | **−0.76** (s.e. 0.31) | A 10% industry-wide price rise cuts real restaurant spending ~7%, but sales still rise ~2% |
| Grocery cross-price elasticity | 0.14 (not significant) | No measurable switching between eating out and groceries |
| Wages → menu prices | 0.52 | A 10% rise in real restaurant wages comes with ~5% higher relative menu prices |

- Demand is **inelastic, but not by much**: price increases raise revenue while losing a meaningful share of traffic.
- This is an **industry-wide** elasticity. A single restaurant raising prices alone faces far more price-sensitive customers.
- The estimate sits inside the 0.7–0.8 range from published research (Andreyeva, Long & Brownell, 2010) and beats simpler models out of sample.

![Prices and spending](output/figures/fig1_prices_and_spending.png)

## Data

All monthly, from FRED (Federal Reserve Bank of St. Louis), January 1992 – August 2026. March 2020 – June 2021 excluded (dining rooms closed).

| Variable | Source | FRED series |
|---|---|---|
| Restaurant sales (food services & drinking places) | Census Bureau | RSFSDP |
| Menu prices (CPI: food away from home) | BLS | CUSR0000SEFV |
| Grocery prices (CPI: food at home) | BLS | CUSR0000SAF11 |
| All-items CPI (to make relative prices) | BLS | CPIAUCSL |
| Real disposable income | BEA | DSPIC96 |
| Restaurant wages (leisure & hospitality hourly earnings) — instrument | BLS | CES7000000008 |
| Food producer prices (PPI processed foods) — instrument | BLS | WPU02 |

Raw file: `data/raw/fredgraph.csv`. Analysis dataset: `data/processed/restaurant_monthly.csv`.

## Method

Relative menu prices trend up steadily, so models in levels depend on how the trend is modeled. The main model uses **12-month changes** (each month vs. the same month a year earlier), which removes trends and seasonality:

```
Δ ln Q_t = a + b·Δ ln P_menu,t + c·Δ ln P_grocery,t + d·Δ ln Income_t + e_t
```

- **Q** = real restaurant spending (sales ÷ menu-price index); prices are relative to the all-items CPI.
- **Two-stage least squares:** menu prices are instrumented with restaurant cost shifters (real wages, real food producer prices), so the estimate reflects customers' response rather than demand shifts. First-stage F = 23.
- **Newey-West** standard errors for overlapping 12-month changes.
- Estimators implemented in NumPy (`src/econometrics.py`) and verified on simulated data (`tests/test_econometrics.py`).

## Validation

1. **Other periods:** 1993–2007 (−0.85) vs 2008–2019 (−0.55), not significantly different; rolling 10-year windows.
2. **Other specifications:** different instrument sets, 6-month changes, levels with and without trends.
3. **Out of sample:** trained on 1993–2019, predicting 2022–2026 spending growth. The price model (RMSE 3.00 pts) beats an income-only model (3.81) and a constant-growth benchmark (3.25).
4. **Benchmark:** Andreyeva, Long & Brownell (2010) report 0.7–0.8.

## Reproduce it

```bash
pip install -r requirements.txt
python tests/test_econometrics.py      # check the estimators
python run_all.py                      # download data, build dataset, estimate, make tables & figures
python run_all.py --no-download        # same, using data/raw/fredgraph.csv already included
```

Every number in the paper comes from the tables in `output/tables/`.

## Repository layout

```
├── run_all.py                  # one command runs the whole pipeline
├── src/
│   ├── 01_download_data.py     # pulls the FRED file into data/raw/
│   ├── 02_build_dataset.py     # cleans, deflates, builds analysis data
│   ├── 03_estimate_models.py   # models, validation, scenarios, figures
│   └── econometrics.py         # OLS / 2SLS with Newey-West SEs
├── tests/test_econometrics.py  # estimators recover known parameters
├── data/raw/, data/processed/
├── output/tables/, output/figures/, output/key_numbers.json
└── paper/                      # the final paper (Word + PDF)
```

## Limitations

National averages only (cannot separate full-service from fast food); wages may partly respond to restaurant demand (the food-cost-only instrument gives a similar but less precise estimate); the post-COVID period is too short to estimate on its own.

## References

- Aaronson, French & MacDonald (2008). The minimum wage, restaurant prices, and labor market structure. *Journal of Human Resources* 43(3), 688–720.
- Andreyeva, Long & Brownell (2010). The impact of food prices on consumption: A systematic review of research on the price elasticity of demand for food. *American Journal of Public Health* 100(2), 216–222.
- Okrent & Alston (2012). The demand for disaggregated food-away-from-home and food-at-home products in the United States. USDA ERS Economic Research Report No. 139.
- Federal Reserve Bank of St. Louis (2026). FRED Economic Data.
- Dong, T. (2026). Supply and demand functions: A survey by data availability. SSRN 7399858.
