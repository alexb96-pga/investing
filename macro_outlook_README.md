# macro_outlook

An exploratory script that asks: **do US macroeconomic indicators say anything about the S&P 500's next-month direction?** (Python script saved without a `.py` extension.)

## What it does

1. Downloads S&P 500 prices (2000 to 2025) and converts them to monthly log returns.
2. Pulls 12 monthly macro series from FRED: Fed funds rate, 10Y yield, CPI, core CPI, unemployment, M2, credit spread, PCE, industrial production, payrolls, housing starts and the 3M T-bill. It adds year-on-year CPI and M2 growth.
3. Scrapes the **Shiller CAPE (P/E)** from multpl.com and adds it as a 13th feature. If the scrape fails, the script continues without it.
4. Prints the **absolute correlation** of each indicator with S&P 500 monthly returns.
5. Computes a **correlation-weighted score**: how far each indicator's latest value is from its average, multiplied by its correlation with returns and summed. A positive score prints an UP signal, otherwise DOWN.
6. Fits a **linear regression** of returns on all indicators and prints the predicted next-month log return with an UP/DOWN signal.

A commented-out block at the bottom plots the rolling 12-month correlation between Shiller PE changes and S&P 500 returns.

## Usage

```bash
pip install pandas numpy yfinance requests fredapi scikit-learn lxml
python macro_outlook
```

You need a FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html), set in the `fred_api_key` variable near the top.

## Limitations

- Indicators are aligned to the same month as the return, with no lag, and the regression is fitted and evaluated on the same data. The "next month" signal is therefore illustrative, not a validated forecast.
- Indicators are used in raw units, so large-magnitude series (payrolls, industrial production) dominate the correlation-weighted score.
- The "BBB Credit Spread" column uses FRED series `BAMLC0A0CM`, the ICE BofA US Corporate (investment grade) spread.
- The end date (`2025-12-01`) is hard-coded.
- Research and education only, not financial advice.
