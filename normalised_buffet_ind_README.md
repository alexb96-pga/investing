# normalised_buffet_ind

Plots a **normalised Buffett Indicator**: a long-run valuation gauge of the US stock market relative to the size of the economy. (Python script saved without a `.py` extension.)

## What it does

1. Downloads the full S&P 500 price history from Yahoo Finance and US nominal GDP from FRED.
2. Interpolates quarterly GDP to daily and divides the S&P 500 level by GDP to get the Buffett Indicator.
3. Smooths it with a 180-day rolling mean.
4. Fits a **linear trend** through the smoothed series and converts the indicator to **standard deviations from that trend** (z-score).
5. Plots the result with lines at the trend (0) and at +/-1 and +/-2 standard deviations. The line is colour-coded from bright red (strongly overvalued) to bright green (strongly undervalued), with labels on each band.
6. Prints the fitted slope, intercept and standard deviation.

Reading the chart: above +2 means the market is historically expensive relative to GDP, below -2 means historically cheap relative to its own trend.

## Usage

```bash
pip install numpy pandas yfinance matplotlib fredapi
python normalised_buffet_ind
```

You need a FRED API key (https://fred.stlouisfed.org/docs/api/api_key.html), set in the `fred_api_key` variable near the top.

## Limitations

- It uses the S&P 500 **price index** as a proxy for total market value (the code comments mention Wilshire 5000, but that is not what is downloaded). The price index excludes dividends and only covers large caps.
- The trend and standard deviation are fitted on the full history, so the bands are in-sample and shift as new data arrives.
- Research and education only, not financial advice.
