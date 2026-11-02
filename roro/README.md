# roro_risk_score.py

Calculates a **Risk-On / Risk-Off (RORO) score** between 0 and 1 from live market and macro data. It is the "Layer 0" risk gauge of a weekly portfolio allocation model, extracted into a standalone module.

- **0.0** = strongly risk-off (stress, tightening financial conditions, weak macro)
- **1.0** = strongly risk-on (calm markets, tight spreads, strong momentum)

## What it does

1. Downloads market data from Yahoo Finance (NASDAQ, VIX, SKEW, copper, DXY, 10Y yield, 3M T-bill) and macro data from FRED (credit spreads, unemployment, industrial production, NFCI).
2. Scores 12 signals, each from 0 to 1, using a 70% level (2-year z-score) and 30% momentum (21/63-day change) blend, squashed through a sigmoid.
3. Takes a weighted average using `SIGNAL_WEIGHTS` to get the raw composite.
4. Applies EMA smoothing and maps the result to a regime label (`RISK-ON (strong)` through `RISK-OFF (strong)`).

| Signal | Source | Direction |
|---|---|---|
| `vix`, `skew` | ^VIX, ^SKEW | higher = risk-off |
| `credit_spread`, `hy_spread` | FRED (Moody's spliced with ICE BofA) | wider = risk-off |
| `nasdaq_primary_momentum` | ^IXIC vs 200-day average | above = risk-on |
| `copper_momentum` | HG=F | rising = risk-on |
| `dxy_momentum` | DX-Y.NYB | stronger dollar = risk-off |
| `yield_curve` | 10Y minus 3M | steeper = risk-on |
| `unemployment`, `industrial_prod` | FRED UNRATE, INDPRO | discounted when data is stale (>45 days) |
| `nfci`, `earnings_rev` | FRED NFCI | looser conditions = risk-on |

Missing data falls back to a neutral 0.5 for that signal rather than failing.

## Usage

```bash
pip install pandas numpy yfinance fredapi
export FRED_API_KEY=your_key_here     # PowerShell: $env:FRED_API_KEY="your_key_here"
python roro_risk_score.py
```

Or from Python:

```python
from roro_risk_score import compute_risk_score
risk_score, raw_score, signals, detail = compute_risk_score()
```

Get a free FRED API key at https://fred.stlouisfed.org/docs/api/api_key.html.

## Notes

- The signal weights are hand-tuned inputs, not guarantees of predictive power.
- The EMA smoothing here runs on a single point, so it returns the raw score. It only smooths if you feed it a history of scores.
- This is a personal research tool, not financial advice.
