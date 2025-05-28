import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def load_yahoo_portfolio(filepath):
    df = pd.read_csv(filepath)
    df = df[df['Quantity'].notna() & (df['Quantity'] > 0)]
    tickers = df['Symbol'].str.upper().tolist()
    shares = df['Quantity'].astype(float).values
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
    mean_returns = daily_returns.mean() * 252
    cov_matrix = daily_returns.cov() * 252
    return mean_returns.values, cov_matrix.values


def generate_recommendations(current_weights, optimized_weights, tickers, prices, portfolio_value, threshold=0.01):
    recs = []
    for ticker, current, optimal, price in zip(tickers, current_weights, optimized_weights, prices):
        delta = optimal - current
        if abs(delta) >= threshold:
            action = "BUY" if delta > 0 else "SELL"
            share_change = int(round(delta * portfolio_value / price))
            if share_change != 0:
                recs.append(f"{action} {abs(share_change)} shares of {ticker}")
    return recs


def efficient_frontier_plot(tickers, mean_returns, cov_matrix, current_weights, prices, portfolio_value):
    results = []
    for _ in range(10000):
        weights = np.random.random(len(tickers))
        weights /= np.sum(weights)
        port_return = np.dot(weights, mean_returns)
        port_stddev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        results.append((port_stddev, port_return, weights))

    results_df = pd.DataFrame(results, columns=['stdev', 'ret', 'weights'])

    max_sharpe = results_df.loc[(results_df['ret'] / results_df['stdev']).idxmax()]
    min_vol = results_df.loc[results_df['stdev'].idxmin()]

    optimized_weights = max_sharpe['weights']
    recs = generate_recommendations(current_weights, optimized_weights, tickers, prices, portfolio_value)

    plt.figure(figsize=(10, 6))
    plt.scatter(results_df['stdev'], results_df['ret'], c=results_df['ret'] / results_df['stdev'], cmap='viridis', alpha=0.5)
    plt.colorbar(label='Sharpe Ratio')
    plt.scatter(max_sharpe['stdev'], max_sharpe['ret'], c='red', marker='*', s=300, label='Max Sharpe')
    plt.scatter(min_vol['stdev'], min_vol['ret'], c='blue', marker='*', s=300, label='Min Volatility')
    plt.xlabel('Volatility (Std Deviation)')
    plt.ylabel('Expected Return')
    plt.title('Efficient Frontier')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('static/efficient_frontier.png')
    plt.close()

    return recs


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            tickers, current_weights, prices, value = load_yahoo_portfolio(filepath)
            price_data = get_data_yfinance(tickers)
            mean_returns, cov_matrix = get_returns_and_cov(price_data)
            recs = efficient_frontier_plot(tickers, mean_returns, cov_matrix, current_weights, prices, value)

            return render_template('results.html', recs=recs, image_path='static/efficient_frontier.png')

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
