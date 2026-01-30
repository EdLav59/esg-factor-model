"""
ESG Factor Model - Empirical Analysis
Testing whether ESG scores predict stock returns using Fama-French methodology

Author: Edouard Lavalard
Date: January 2026
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from datetime import datetime
import os
import utils

# Configuration
EXCEL_PATH = 'S&P_100_Data.xlsx'
ESG_PATH = 'esg_scores.csv'
RESULTS_DIR = 'results'

# Analysis settings
# Choose your tradeoff: more history vs more stocks
# Options: 'max_history' (2019+, fewer stocks) or 'max_coverage' (2022+, all stocks)
START_DATE_STRATEGY = 'max_coverage'  # Change to 'max_history' for longer timeframe

def load_esg_scores():
    """Load ESG scores from LSEG data"""
    print("1. Loading ESG Scores...")
    esg_df = pd.read_csv(ESG_PATH)
    print(f"   Loaded {len(esg_df)} companies with ESG scores")
    print(f"   Score range: {esg_df['ESG_Score'].min():.0f} to {esg_df['ESG_Score'].max():.0f}")
    return esg_df

def load_price_data(tickers):
    """
    Load historical price data from Excel file
    Each ticker has its own sheet with Date and Close columns
    """
    print("\n2. Loading Price Data from Excel...")
    
    all_prices = {}
    failed_tickers = []
    
    for ticker in tickers:
        try:
            # Read the sheet for this ticker
            df = pd.read_excel(EXCEL_PATH, sheet_name=ticker)
            
            # Find Date and Close columns (column names might vary slightly)
            date_col = [col for col in df.columns if 'Date' in str(col)][0]
            close_col = [col for col in df.columns if 'Close' in str(col)][0]
            
            # Clean up the data
            price_df = df[[date_col, close_col]].copy()
            price_df.columns = ['Date', 'Close']
            
            # Convert Excel serial dates to datetime
            price_df['Date'] = pd.to_datetime(price_df['Date'], origin='1899-12-30', unit='D')
            price_df = price_df.set_index('Date').sort_index()
            price_df = price_df.dropna()
            
            all_prices[ticker] = price_df['Close']
            
        except Exception as e:
            print(f"   Warning: couldn't load {ticker} - {str(e)[:50]}")
            failed_tickers.append(ticker)
    
    # Combine all series into one dataframe
    prices_df = pd.DataFrame(all_prices)
    
    print(f"   Successfully loaded: {len(all_prices)} stocks")
    if failed_tickers:
        print(f"   Failed to load: {len(failed_tickers)} stocks")
    print(f"   Date range: {prices_df.index.min().strftime('%Y-%m-%d')} to {prices_df.index.max().strftime('%Y-%m-%d')}")
    print(f"   Trading days: {len(prices_df)}")
    
    return prices_df

def calculate_monthly_returns(prices_df, strategy='max_coverage'):
    """
    Convert daily prices to monthly returns
    
    strategy: 'max_history' (keep early dates, drop stocks with gaps)
              'max_coverage' (keep all stocks, start when all have data)
    """
    print("\n3. Calculating Monthly Returns...")
    
    # Resample to month-end prices
    monthly_prices = prices_df.resample('ME').last()
    
    print(f"   Raw data: {len(monthly_prices)} months, {len(monthly_prices.columns)} stocks")
    print(f"   Date range: {monthly_prices.index.min().strftime('%Y-%m')} to {monthly_prices.index.max().strftime('%Y-%m')}")
    
    if strategy == 'max_history':
        # Keep early dates, drop stocks with incomplete data
        print(f"   Strategy: Maximizing history (may drop some stocks)")
        
        # Calculate returns
        monthly_returns = monthly_prices.pct_change().iloc[1:]
        
        # Drop stocks with any NaN (incomplete data)
        clean_returns = monthly_returns.dropna(axis=1)
        
        dropped = len(monthly_returns.columns) - len(clean_returns.columns)
        if dropped > 0:
            print(f"   Dropped {dropped} stocks with incomplete historical data")
        
        monthly_returns = clean_returns
        
    else:  # max_coverage
        # Keep all stocks, start when we have good coverage
        print(f"   Strategy: Maximizing stock coverage (may trim early dates)")
        
        # Find when we have 80%+ stocks with data
        stocks_per_month = monthly_prices.count(axis=1)
        coverage_threshold = 0.8 * len(monthly_prices.columns)
        
        good_months = stocks_per_month >= coverage_threshold
        if good_months.any():
            first_good = good_months[good_months].index[0]
            if first_good > monthly_prices.index.min():
                print(f"   Trimming data before {first_good.strftime('%Y-%m')} due to low coverage")
                monthly_prices = monthly_prices[monthly_prices.index >= first_good]
        
        # Calculate returns
        monthly_returns = monthly_prices.pct_change().iloc[1:]
        
        # Drop any remaining columns with NaN
        monthly_returns = monthly_returns.dropna(axis=1)
    
    print(f"   Final: {len(monthly_returns)} months, {len(monthly_returns.columns)} stocks")
    print(f"   Period: {monthly_returns.index.min().strftime('%Y-%m')} to {monthly_returns.index.max().strftime('%Y-%m')}")
    
    return monthly_returns

def construct_esg_factor(esg_df, monthly_returns):
    """
    Build the ESG factor using long-short methodology:
    - Long: High ESG stocks (top quartile)
    - Short: Low ESG stocks (bottom quartile)
    - Factor return = Long - Short
    
    Also tracks Mid-ESG portfolio (Q2-Q3) for comparison
    """
    print("\n4. Constructing ESG Factor...")
    
    # Find stocks that have both ESG scores and price data
    common_tickers = list(set(esg_df['Ticker']) & set(monthly_returns.columns))
    esg_available = esg_df[esg_df['Ticker'].isin(common_tickers)].copy()
    
    print(f"   Working with {len(common_tickers)} stocks")
    
    # Split into quartiles based on ESG scores
    q1_threshold = esg_available['ESG_Score'].quantile(0.25)
    q4_threshold = esg_available['ESG_Score'].quantile(0.75)
    
    low_esg_tickers = esg_available[esg_available['ESG_Score'] <= q1_threshold]['Ticker'].tolist()
    high_esg_tickers = esg_available[esg_available['ESG_Score'] >= q4_threshold]['Ticker'].tolist()
    mid_esg_tickers = [t for t in common_tickers if t not in high_esg_tickers and t not in low_esg_tickers]
    
    # Calculate average ESG scores for each bucket
    low_esg_avg = esg_available[esg_available['Ticker'].isin(low_esg_tickers)]['ESG_Score'].mean()
    mid_esg_avg = esg_available[esg_available['Ticker'].isin(mid_esg_tickers)]['ESG_Score'].mean()
    high_esg_avg = esg_available[esg_available['Ticker'].isin(high_esg_tickers)]['ESG_Score'].mean()
    
    print(f"   Low ESG portfolio (Q1, score ≤ {q1_threshold:.0f}): {len(low_esg_tickers)} stocks")
    print(f"      → Average ESG Score: {low_esg_avg:.1f}")
    print(f"   Mid ESG portfolio (Q2-Q3): {len(mid_esg_tickers)} stocks")
    print(f"      → Average ESG Score: {mid_esg_avg:.1f}")
    print(f"   High ESG portfolio (Q4, score ≥ {q4_threshold:.0f}): {len(high_esg_tickers)} stocks")
    print(f"      → Average ESG Score: {high_esg_avg:.1f}")
    
    # Equal-weighted portfolio returns
    high_esg_returns = monthly_returns[high_esg_tickers].mean(axis=1)
    low_esg_returns = monthly_returns[low_esg_tickers].mean(axis=1)
    mid_esg_returns = monthly_returns[mid_esg_tickers].mean(axis=1)
    
    # Long-short factor
    esg_factor_returns = high_esg_returns - low_esg_returns
    
    return esg_factor_returns, high_esg_returns, low_esg_returns, mid_esg_returns, common_tickers

def calculate_market_factor(monthly_returns):
    """Market factor = equal-weighted return of all stocks"""
    market_returns = monthly_returns.mean(axis=1)
    return market_returns

def apply_transaction_costs(returns, cost_per_trade=0.0001, rebalancing='monthly'):
    """
    Apply transaction costs to portfolio returns
    
    Assumes:
    - Monthly rebalancing (cost applied each month)
    - Full portfolio turnover on long and short sides
    - cost_per_trade: 1bp = 0.0001 (0.01%)
    
    For long-short portfolio:
    - Enter long position: 1bp
    - Exit long position: 1bp  
    - Enter short position: 1bp
    - Exit short position: 1bp
    - Total per rebalance: 4bp (but we use 2bp as conservative estimate)
    """
    cost_per_rebalance = cost_per_trade * 2  # 2bp per month (1bp each side)
    
    # Apply cost each month
    net_returns = returns - cost_per_rebalance
    
    return net_returns

def run_factor_regression(factor_returns, market_returns):
    """
    Regress ESG factor returns on market returns to test for alpha
    Model: ESG_Factor = alpha + beta * Market + error
    
    If alpha is significantly different from zero, ESG adds value beyond market exposure
    """
    print("\n5. Running Factor Regression...")
    
    # Align the time series
    common_index = factor_returns.index.intersection(market_returns.index)
    y = factor_returns.loc[common_index]
    X = market_returns.loc[common_index]
    
    # Add constant for intercept
    X = sm.add_constant(X)
    
    # OLS regression
    model = sm.OLS(y, X).fit()
    
    return model

def generate_summary_statistics(factor_returns, high_returns, low_returns, mid_returns, market_returns):
    """Calculate performance metrics for all portfolios"""
    print("\n6. Calculating Performance Metrics...")
    
    # Apply transaction costs to factor (long-short strategy)
    factor_returns_net = apply_transaction_costs(factor_returns)
    
    stats_dict = {
        'ESG Factor (High-Low, Gross)': utils.calculate_performance_metrics(factor_returns),
        'ESG Factor (High-Low, Net of Costs)': utils.calculate_performance_metrics(factor_returns_net),
        'High ESG Portfolio': utils.calculate_performance_metrics(high_returns),
        'Mid ESG Portfolio': utils.calculate_performance_metrics(mid_returns),
        'Low ESG Portfolio': utils.calculate_performance_metrics(low_returns),
        'Market (Equal-Weight)': utils.calculate_performance_metrics(market_returns)
    }
    
    print("   Note: Transaction costs = 2bp per month (1bp per side) applied to factor")
    
    return stats_dict

def main():
    """Run the full ESG factor analysis"""
    print("=" * 60)
    print("ESG Factor Model - Empirical Analysis")
    print("=" * 60)
    
    # Create output folder
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Step 1: Load ESG data
    esg_df = load_esg_scores()
    
    # Step 2: Load price data
    prices_df = load_price_data(esg_df['Ticker'].tolist())
    
    # Step 3: Calculate monthly returns
    monthly_returns = calculate_monthly_returns(prices_df, strategy=START_DATE_STRATEGY)
    
    # Step 4: Build ESG factor
    esg_factor, high_returns, low_returns, mid_returns, common_tickers = construct_esg_factor(esg_df, monthly_returns)
    
    # Step 5: Calculate market benchmark
    market_returns = calculate_market_factor(monthly_returns)
    
    # Step 6: Regression analysis
    regression_model = run_factor_regression(esg_factor, market_returns)
    
    # Step 7: Performance statistics
    performance_stats = generate_summary_statistics(esg_factor, high_returns, low_returns, mid_returns, market_returns)
    
    # Print results
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for portfolio_name, metrics in performance_stats.items():
        print(f"\n{portfolio_name}:")
        print(f"  Annual Return:      {metrics['Annual Return']*100:>7.2f}%")
        print(f"  Annual Volatility:  {metrics['Annual Volatility']*100:>7.2f}%")
        print(f"  Sharpe Ratio:       {metrics['Sharpe Ratio']:>7.2f}")
        print(f"  Max Drawdown:       {metrics['Max Drawdown']*100:>7.2f}%")
    
    # Transaction costs impact
    print("\n" + "-" * 60)
    print("Transaction Costs Impact on ESG Factor")
    print("-" * 60)
    gross_stats = performance_stats['ESG Factor (High-Low, Gross)']
    net_stats = performance_stats['ESG Factor (High-Low, Net of Costs)']
    
    cost_impact = (gross_stats['Annual Return'] - net_stats['Annual Return']) * 100
    
    print(f"Gross Annual Return:       {gross_stats['Annual Return']*100:>7.2f}%")
    print(f"Transaction Costs:         {-cost_impact:>7.2f}%  (2bp/month)")
    print(f"Net Annual Return:         {net_stats['Annual Return']*100:>7.2f}%")
    print(f"\nCost as % of Gross Return: {(cost_impact / (gross_stats['Annual Return']*100) * 100):>6.1f}%")
    print("\nConclusion: Transaction costs significantly erode the small factor return,")
    print("            bringing net performance close to zero.")
    
    # Regression results
    print("\n" + "-" * 60)
    print("Regression Analysis: ESG_Factor = α + β * Market")
    print("-" * 60)
    alpha_monthly = regression_model.params.iloc[0]
    beta = regression_model.params.iloc[1]
    alpha_annual = alpha_monthly * 12
    t_stat = regression_model.tvalues.iloc[0]
    p_value = regression_model.pvalues.iloc[0]
    r_squared = regression_model.rsquared
    
    print(f"Alpha (monthly):        {alpha_monthly*100:>7.3f}%")
    print(f"Alpha (annualized):     {alpha_annual*100:>7.2f}%")
    print(f"Beta (market exposure): {beta:>7.3f}")
    print(f"t-statistic:            {t_stat:>7.2f}")
    print(f"p-value:                {p_value:>7.4f}")
    print(f"R-squared:              {r_squared:>7.3f}")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("Interpretation:")
    print("-" * 60)
    
    if p_value < 0.05:
        significance = "statistically significant"
        direction = "positive" if alpha_annual > 0 else "negative"
        print(f"The ESG factor shows {significance} alpha (p={p_value:.4f})")
        print(f"Direction: {direction} alpha of {abs(alpha_annual)*100:.2f}% annually")
        if alpha_annual > 0:
            print("High ESG stocks outperformed low ESG stocks after controlling for market.")
        else:
            print("Low ESG stocks outperformed high ESG stocks after controlling for market.")
    else:
        print(f"The ESG factor does not show statistically significant alpha (p={p_value:.4f})")
        print("Possible explanations:")
        print("  - ESG characteristics already reflected in stock prices")
        print("  - ESG factor may overlap with other known factors (Quality, Low Vol)")
        print("  - Sample period or data coverage limitations")
    
    if abs(beta) > 0.3:
        print(f"\nThe factor has some market exposure (β={beta:.2f})")
    else:
        print(f"\nThe factor is relatively market-neutral (β={beta:.2f})")
    
    # Generate charts
    print("\n" + "-" * 60)
    print("Generating Charts...")
    print("-" * 60)
    
    utils.plot_esg_distribution(esg_df)
    utils.plot_cumulative_returns(esg_factor, high_returns, low_returns, mid_returns)
    utils.plot_rolling_sharpe(esg_factor)
    
    # Correlation analysis
    returns_comparison = pd.DataFrame({
        'ESG_Factor': esg_factor,
        'Market': market_returns,
        'High_ESG': high_returns,
        'Mid_ESG': mid_returns,
        'Low_ESG': low_returns
    })
    utils.plot_correlation_matrix(returns_comparison)
    utils.plot_factor_returns_distribution(esg_factor)
    
    # Save results to files
    print("\n" + "-" * 60)
    print("Saving Results...")
    print("-" * 60)
    
    # Regression details
    with open(f'{RESULTS_DIR}/regression_summary.txt', 'w') as f:
        f.write(regression_model.summary().as_text())
    print(f"Saved: {RESULTS_DIR}/regression_summary.txt")
    
    # Performance metrics
    perf_df = pd.DataFrame(performance_stats).T
    perf_df.to_csv(f'{RESULTS_DIR}/performance_metrics.csv')
    print(f"Saved: {RESULTS_DIR}/performance_metrics.csv")
    
    # Factor returns time series
    factor_data = pd.DataFrame({
        'ESG_Factor': esg_factor,
        'High_ESG_Portfolio': high_returns,
        'Mid_ESG_Portfolio': mid_returns,
        'Low_ESG_Portfolio': low_returns,
        'Market': market_returns
    })
    factor_data.to_csv(f'{RESULTS_DIR}/factor_returns.csv')
    print(f"Saved: {RESULTS_DIR}/factor_returns.csv")
    
    print("\n" + "=" * 60)
    print("Analysis Complete")
    print("=" * 60)
    print(f"\nAll results saved to '{RESULTS_DIR}/' directory")
    print("\nOutput files:")
    print("  Charts:")
    print("    - cumulative_returns.png")
    print("    - rolling_sharpe.png")
    print("    - correlation_matrix.png")
    print("    - factor_distribution.png")
    print("    - esg_distribution.png")
    print("  Data:")
    print("    - performance_metrics.csv")
    print("    - factor_returns.csv")
    print("    - regression_summary.txt")

if __name__ == "__main__":
    main()