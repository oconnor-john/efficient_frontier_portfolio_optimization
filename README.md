# Yahoo Finance Portfolio Optimizer

This Python tool loads your exported Yahoo Finance portfolio, fetches historical price data, runs a portfolio optimization using the efficient frontier, and generates **buy/sell trade recommendations in number of shares** to optimize your Sharpe ratio.

Note: The provided portfolio.csv is for example use only.

---

## Features

- 📥 Load portfolio directly from a Yahoo Finance `.csv` export
- 📈 Download historical adjusted price data using `yfinance`
- 🧠 Use Monte Carlo simulation to generate the efficient frontier
- ⭐ Maximize Sharpe ratio or minimize volatility
- 🛠️ Get **clear BUY/SELL trade suggestions in shares**
- 📉 Visualize the efficient frontier with matplotlib

---

## Requirements

Make sure you have Python 3.7+ installed, then run:

```bash
pip install numpy pandas matplotlib yfinance
