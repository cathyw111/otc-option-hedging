# OTC Option Hedging Costs and Quote Add-ons

A simulation study of how transaction costs and discrete delta hedging affect the extra premium required to cover a short European call. The implementation compares frequent rebalancing, fixed intervals and delta bands under ETF and stock cost assumptions.

## Main result

Lower trading costs do not necessarily imply a lower tail-risk add-on. In the supplied 3-month ATM ETF simulation, trading every five days reduces mean fees but increases the 95% loss-covering add-on relative to four trades per day.

| Rule | Mean trading fees (bp) | Mean-loss add-on (bp) | 95% add-on (bp) |
| --- | ---: | ---: | ---: |
| Four times per day | 19.58 | 21.30 | 58.55 |
| Once per day | 10.82 | 11.69 | 82.20 |
| Every five days | 6.08 | 5.22 | 148.47 |
| Delta band 0.06 | 8.40 | 9.58 | 90.63 |

Source: [supplied ETF results](out/rules_etf.csv), 4,000 simulated paths, CNY 10 million notional. One basis point is 0.01% of notional. These are simulated results under configured cost assumptions, rather than observed market quotes.

## Implementation

- [bs.py](src/bs.py): vectorized Black-Scholes call price, Greeks and implied volatility.
- [hedge.py](src/hedge.py): GBM and GJR-GARCH paths, block bootstrap helper, cash-account financing, lot rounding and transaction costs.
- [config.py](src/config.py): notional, volatility, cost rates, maturities and random seed.
- `seg2_baseline.py` through `seg6_deliver.py`: baseline checks, cost comparisons, rebalancing rules, model sensitivity and quote tables.

For path profit `pnl`, the add-on is `max(-quantile(pnl, 0.05), 0) / notional * 10_000`. The current cash ledger adds the initial premium to terminal hedge profit without accruing that premium; interpret results using this convention.

## Run

Python 3.12 is the reference version indicated by the supplied code package. From the repository root:

```bash
python -m pip install -r requirements.txt
cd src
python seg2_baseline.py
python seg3_cost.py
python seg4_rules.py
python seg5_robust.py
python seg6_deliver.py
```

Scripts write to the existing `out/` directory and use independent seeded generators. Stages 5 and 6 perform bootstrap resampling and can take longer than the baseline. No network connection or private market data is required. The source package mentions `seg1_data.py`, but does not contain that script; no market-data ingestion is claimed here.

## Results and interpretation

The six supplied CSV files in [out](out) cover ETF/stock rule comparisons, model sensitivity, bootstrap intervals, maturity/moneyness quote add-ons and a band grid. Column labels are retained in their original Chinese form. `手续费bp` means trading fees in basis points; `加点95` means the 95% loss-covering add-on; `调仓次数` means the number of nonzero trades. `简单` denotes GBM and `聚集` denotes the GJR model.

Bootstrap intervals resample simulated paths; they are not confidence intervals from 250 actual market windows. GJR and GBM comparisons also use different drift conventions in the supplied implementation. The band table reports the best point estimate on a finite grid, without establishing that nearby bands are statistically distinguishable. Cost rates are experimental parameters, not a statement of current exchange rules.
