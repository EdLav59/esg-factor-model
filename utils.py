"""
Helper functions for ESG Factor Model
Performance metrics, plotting, and analysis utilities
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def calculate_performance_metrics(returns):
    """
    Calculate standard performance metrics from a return series
    Assumes monthly returns
    """
    ann_factor = 12  # annualization factor for monthly data
    
    mean_ret = returns.mean() * ann_factor
    volatility = returns.std() * np.sqrt(ann_factor)
    sharpe = mean_ret / volatility if volatility != 0 else 0
    
    # Cumulative return over the period
    cum_ret = (1 + returns).prod() - 1
    
    # Maximum drawdown calculation
    cum_returns = (1 + returns).cumprod()
    running_max = cum_returns.expanding().max()
    drawdown = (cum_returns - running_max) / running_max
    max_dd = drawdown.min()
    
    return {
        'Annual Return': mean_ret,
        'Annual Volatility': volatility,
        'Sharpe Ratio': sharpe,
        'Cumulative Return': cum_ret,
        'Max Drawdown': max_dd,
        'Total Periods': len(returns)
    }

def plot_cumulative_returns(factor_returns, high_ret, low_ret, mid_ret=None, save_path='results/cumulative_returns.png'):
    """Plot cumulative performance of ESG factor and its components"""
    try:
        # Ensure output directory exists
        import os
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Calculate cumulative returns
        cum_factor = (1 + factor_returns).cumprod() - 1
        cum_high = (1 + high_ret).cumprod() - 1
        cum_low = (1 + low_ret).cumprod() - 1
        
        # Main factor line
        ax.plot(cum_factor.index, cum_factor * 100, 
                label='ESG Factor (High - Low)', linewidth=2.5, color='green')
        
        # Component portfolios
        ax.plot(cum_high.index, cum_high * 100, 
                label='High ESG Portfolio (Q4)', linewidth=1.8, alpha=0.8, 
                color='darkgreen', linestyle='--')
        
        # Add Mid-ESG if provided
        if mid_ret is not None:
            cum_mid = (1 + mid_ret).cumprod() - 1
            ax.plot(cum_mid.index, cum_mid * 100, 
                    label='Mid ESG Portfolio (Q2-Q3)', linewidth=1.8, alpha=0.8, 
                    color='blue', linestyle='-.')
        
        ax.plot(cum_low.index, cum_low * 100, 
                label='Low ESG Portfolio (Q1)', linewidth=1.8, alpha=0.8, 
                color='red', linestyle='--')
        
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
        ax.set_title('ESG Factor: Cumulative Returns by ESG Quartile', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Cumulative Return (%)', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)  # Important: close figure to free memory
        print(f"Saved: {save_path}")
        
    except Exception as e:
        print(f"Error creating cumulative returns plot: {e}")
        import traceback
        traceback.print_exc()

def plot_rolling_sharpe(factor_returns, window=12, save_path='results/rolling_sharpe.png'):
    """Plot rolling Sharpe ratio to see time variation in risk-adjusted returns"""
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate rolling metrics
    rolling_mean = factor_returns.rolling(window).mean() * 12
    rolling_std = factor_returns.rolling(window).std() * np.sqrt(12)
    rolling_sharpe = rolling_mean / rolling_std
    
    ax.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='steelblue')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axhline(y=1, color='green', linestyle='--', linewidth=1, 
               alpha=0.5, label='Sharpe = 1')
    
    ax.set_title(f'ESG Factor: Rolling Sharpe Ratio ({window}-Month Window)', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Sharpe Ratio', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_correlation_matrix(returns_df, save_path='results/correlation_matrix.png'):
    """Plot correlation heatmap between different return series"""
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    corr = returns_df.corr()
    
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdYlGn', center=0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1, ax=ax)
    
    ax.set_title('Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_esg_distribution(esg_scores, save_path='results/esg_distribution.png'):
    """Plot distribution of ESG scores with quartile markers"""
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    scores = esg_scores['ESG_Score'].values
    
    # Histogram
    ax.hist(scores, bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    
    # Add quartile lines
    q1 = np.percentile(scores, 25)
    q2 = np.percentile(scores, 50)
    q3 = np.percentile(scores, 75)
    
    ax.axvline(q1, color='red', linestyle='--', linewidth=2, 
               label=f'Q1 (25th): {q1:.0f}')
    ax.axvline(q2, color='orange', linestyle='--', linewidth=2, 
               label=f'Q2 (50th): {q2:.0f}')
    ax.axvline(q3, color='green', linestyle='--', linewidth=2, 
               label=f'Q3 (75th): {q3:.0f}')
    
    ax.set_title('ESG Score Distribution (LSEG Data)', fontsize=14, fontweight='bold')
    ax.set_xlabel('ESG Score', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")

def plot_factor_returns_distribution(factor_returns, save_path='results/factor_distribution.png'):
    """Plot distribution of factor returns and check for normality"""
    # Ensure output directory exists
    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of returns
    ax1.hist(factor_returns * 100, bins=30, alpha=0.7, 
             color='steelblue', edgecolor='black')
    ax1.axvline(0, color='red', linestyle='--', linewidth=2)
    ax1.set_title('ESG Factor Returns Distribution', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Monthly Return (%)', fontsize=10)
    ax1.set_ylabel('Frequency', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Q-Q plot to check normality
    from scipy import stats
    stats.probplot(factor_returns, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")