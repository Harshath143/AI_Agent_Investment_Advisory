import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# Parameters for Synthethic Data Generation

np.random.seed(42)  # For reproducibility
n_days = 365 * 2 # 2 years of daily data
n_stocks = 10  # Number of stocks
n_users = 1000  # Number of users
tickers =  ['AAPL', 'TSLA', 'AMZN', 'GOOGL', 'MSFT', 'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'WIPRO']

# Function to generate synthetic stock prices using Geometric Brownian Motion

def generate_gbm_prices(S0, mu, sigma, days):
    dt = 1 / 252  # Trading days in a year
    price_series = [S0]
    for _ in range(days - 1):
        dW = np.random.normal(0, np.sqrt(dt))
        St = price_series[-1] * np.exp((mu - 0.5 * sigma**2) * dt + sigma * dW)
        price_series.append(St)
    return np.array(price_series)

# Generate stock price data

stock_data = {}
for ticker in tickers:
    stock_data[ticker] = generate_gbm_prices(S0=np.random.uniform(100, 300), mu=np.random.uniform(0.05, 0.15), sigma=np.random.uniform(0.2, 0.4), days=n_days)

df_stocks = pd.DataFrame(stock_data, index=pd.date_range(start='2023-01-01', periods=n_days))
df_stocks.index.name = 'Date'

df_stocks.to_csv("synthetic_stock_prices.csv")

# Simulating Mutual Fund NAVs with volatility clustering

def simulate_mutual_fund_nav(days, base_value=100, volatility=0.02):
    returns = np.random.normal(0, volatility, days)
    nav_series = base_value * np.cumprod(1 + returns)
    return nav_series

df_mutual_funds = pd.DataFrame({
    "Fund_1": simulate_mutual_fund_nav(n_days, volatility=0.02),
    "Fund_2": simulate_mutual_fund_nav(n_days, volatility=0.018),
    "Fund_3": simulate_mutual_fund_nav(n_days, volatility=0.025),
    "Fund_4": simulate_mutual_fund_nav(n_days, volatility=0.015),
    "Fund_5": simulate_mutual_fund_nav(n_days, volatility=0.022),
    "Fund_6": simulate_mutual_fund_nav(n_days, volatility=0.017),
    "Fund_7": simulate_mutual_fund_nav(n_days, volatility=0.023),
    "Fund_8": simulate_mutual_fund_nav(n_days, volatility=0.019),
    "Fund_9": simulate_mutual_fund_nav(n_days, volatility=0.021),
    "Fund_10": simulate_mutual_fund_nav(n_days, volatility=0.016)
}, index=pd.date_range(start='2023-01-01', periods=n_days))
df_mutual_funds.index.name = 'Date'

df_mutual_funds.to_csv("synthetic_mutual_funds.csv")

# Generating user risk profiles

risk_profiles = np.random.choice(['Low', 'Medium', 'High'], size=n_users, p=[0.4, 0.4, 0.2])
user_data = pd.DataFrame({
    "User_ID": [f"U{str(i).zfill(4)}" for i in range(1, n_users + 1)],
    "Risk_Profile": risk_profiles
})
user_data.to_csv("synthetic_user_profiles.csv", index=False)

print("Synthetic financial data generated and saved as CSV files.")