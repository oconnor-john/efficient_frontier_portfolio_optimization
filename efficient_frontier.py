import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

def load_yahoo_portfolio(filepath="portfolio.csv"):
    df = pd.read_csv(filepath)
    df = df[df['Quantity'].notna()]
    df = df[df['Quantity'] > 0]

    tickers = df['Symbol'].str.upper().tolist()
    shares = df['Quantity'].astype(float).values

    print("Tickers being downloaded:", tickers)

    data = yf.download(tickers, period='1d', progress=False, auto_adjust=False)
    prices = data['Adj Close'].iloc[-1]

    values = shares * prices.values
    weights = values / np.sum(values)

    return tickers, weights, prices.values, np.sum(values)

def get_data_yfinance(tickers, start='2023-01-01', end='2024-01-01'):
    data = yf.download(tickers, start=start, end=end, group_by='ticker', auto_adjust=False)
    if isinstance(data.columns, pd.MultiIndex):
        adj_close = data.loc[:, pd.IndexSlice[:, 'Adj Close']]
        adj_close.columns = adj_close.columns.droplevel(1)
    else:
        adj_close = data['Adj Close'].to_frame()
    return adj_close

def get_returns_and_cov(data):
    daily_returns = data.pct_change().dropna()
    mean_returns = daily_returns.mean() * 252  # annualized
    cov_matrix = daily_returns.cov() * 252     # annualized
    return mean_returns.values, cov_matrix.values

def generate_trade_recommendations(current_weights, optimized_weights, tickers, prices, portfolio_value, threshold=0.01):
    print("\n--- Trading Recommendations ---")
    for ticker, current, optimal, price in zip(tickers, current_weights, optimized_weights, prices):
        delta = optimal - current
        if abs(delta) >= threshold:
            action = "BUY" if delta > 0 else "SELL"
            dollar_change = delta * portfolio_value
            share_change = int(round(dollar_change / price))
            if share_change != 0:
                print(f"{action} {abs(share_change)} shares of {ticker}")
    print("--- End of Recommendations ---")

def efficient_frontier_analysis(tickers, mean_returns, cov_matrix,
                                 current_weights=None, prices=None, portfolio_value=None,
                                 num_portfolios=10000, benchmark_ticker='SPY'):

    results = np.zeros((3, num_portfolios))
    weights_record = []

    for _ in range(num_portfolios):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        weights_record.append(weights)

        port_return = np.dot(weights, mean_returns)
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        sharpe = port_return / port_stddev

        results[0, _] = port_return
        results[1, _] = port_stddev
        results[2, _] = sharpe

    results_df = pd.DataFrame(results.T, columns=['ret', 'stdev', 'sharpe'])
    weights_df = pd.DataFrame(weights_record, columns=tickers)
    results_df = pd.concat([results_df, weights_df], axis=1)

    max_sharpe_port = results_df.iloc[results_df['sharpe'].idxmax()]
    min_vol_port = results_df.iloc[results_df['stdev'].idxmin()]

    # Benchmark to SPY
    try:
        print(f"Fetching benchmark data: {benchmark_ticker}...")
        benchmark_raw = yf.download(benchmark_ticker, start='2023-01-01', end='2024-01-01', progress=False, auto_adjust=False)

        if 'Adj Close' in benchmark_raw.columns:
            benchmark_data = benchmark_raw['Adj Close']
        elif isinstance(benchmark_raw.columns, pd.MultiIndex):
            benchmark_data = benchmark_raw.loc[:, pd.IndexSlice[:, 'Adj Close']].droplevel(1, axis=1).iloc[:, 0]
        else:
            raise ValueError("Could not find 'Adj Close' in benchmark data.")

        benchmark_returns = benchmark_data.pct_change().dropna()
        bench_annual_return = benchmark_returns.mean() * 252
        bench_annual_vol = benchmark_returns.std() * np.sqrt(252)

    except Exception as e:
        print("Failed to fetch benchmark:", e)
        bench_annual_return = None
        bench_annual_vol = None

    # Plot efficient frontier and benchmark
    plt.figure(figsize=(10, 7))
    plt.scatter(results_df.stdev, results_df.ret, c=results_df.sharpe, cmap='viridis', alpha=0.7)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe_port['stdev'], max_sharpe_port['ret'], c='red', marker='*', s=300, label='Max Sharpe')
    plt.scatter(min_vol_port['stdev'], min_vol_port['ret'], c='blue', marker='*', s=300, label='Min Volatility')

    if bench_annual_return is not None and bench_annual_vol is not None:
        plt.scatter(bench_annual_vol, bench_annual_return, c='orange', marker='X', s=200, label=f'Benchmark ({benchmark_ticker})')

    plt.xlabel('Volatility (Std Deviation)')
    plt.ylabel('Return')
    plt.title('Efficient Frontier with Benchmark')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Generate trade recommendations
    if current_weights is not None and prices is not None and portfolio_value is not None:
        optimized_weights = max_sharpe_port[tickers].values
        generate_trade_recommendations(current_weights, optimized_weights, tickers, prices, portfolio_value)

    return {
        "max_sharpe_portfolio": max_sharpe_port,
        "min_vol_portfolio": min_vol_port,
        "all_portfolios": results_df
    }

if __name__ == "__main__":
    try:
        print("Loading portfolio from Yahoo export (portfolio.csv)...")
        tickers, current_weights, latest_prices, portfolio_value = load_yahoo_portfolio("portfolio.csv")

        print("Fetching historical price data...")
        price_data = get_data_yfinance(tickers, start='2023-01-01', end='2024-01-01')
        mean_returns, cov_matrix = get_returns_and_cov(price_data)

        print("Running portfolio optimization and generating recommendations...")
        efficient_frontier_analysis(tickers, mean_returns, cov_matrix, current_weights, latest_prices, portfolio_value)

    except Exception as e:
        print("ERROR:", e)
        print("Make sure 'portfolio.csv' is in the same folder and contains 'Symbol' and 'Quantity' columns.")
