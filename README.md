# ESG as a Factor: Empirical Analysis

## Research Question
Do ESG scores have predictive power for stock returns?

This project tests whether ESG ratings function as a return-generating factor, similar to traditional factors like value or momentum.

## Key Finding: Non-Linear ESG Relationship

The analysis reveals an inverted-U pattern - mid-range ESG stocks significantly outperformed both extremes:

| Portfolio | Stocks | Avg ESG | Return | Sharpe |
|-----------|--------|---------|--------|--------|
| **Mid ESG (Q2-Q3)** | 45 | 72.7 | **17.8%** | 1.02 |
| Market (All) | 100 | 71.3 | 15.2% | 0.89 |
| High ESG (Q4) | 30 | 83.2 | 13.3% | 0.75 |
| Low ESG (Q1) | 25 | 57.4 | 13.0% | 0.74 |

The long-short ESG factor (High minus Low) generated only 0.3% annually before costs and 0.07% after transaction costs so, economically zero.

## Interpretation

The non-linear relationship suggests:

1. **ESG is priced efficiently** - no systematic mispricing to exploit
2. **Sweet spot exists** - mid-range ESG (score ~73) offers optimal risk/return
3. **Extremes underperform** for different reasons:
   - Low ESG: Headline risk, regulatory concerns
   - High ESG: Premium already baked into valuations
4. **Transaction costs matter** - 2bp/month erodes 77% of gross returns

This contrasts with simple narratives of an "ESG premium" and highlights the importance of empirical testing.

## Methodology

**Universe**: S&P 100 stocks (2019-2025)

**ESG Data**: LSEG (London Stock Exchange Group) scores on a 0-100 scale

Note: ESG scores vary by provider due to different methodologies. LSEG emphasizes disclosure quality and materiality assessment. Results may differ with other providers (MSCI, Sustainalytics, etc.).

**Factor Construction**:
1. Rank stocks by ESG score
2. Split into quartiles (Q1, Q2, Q3, Q4)
3. Long Q4 (high ESG), short Q1 (low ESG)
4. Equal-weighted, monthly rebalancing

**Analysis**:
- Performance metrics (return, volatility, Sharpe, drawdown)
- Regression: `ESG_Factor = α + β × Market`
- Test alpha significance (t-test, p-value)
- Transaction costs: 2bp per month (1bp each side)

## Results Summary

### Performance (Feb 2019 - Dec 2025, 83 months)

**ESG Factor:**
- Gross Return: 0.31% annually
- Net Return (after costs): 0.07% annually
- Sharpe Ratio: 0.01 (net)
- Market Beta: 0.04 (market-neutral)

**Statistical Test:**
- Alpha: -0.30% annualized
- t-statistic: -0.12
- p-value: 0.91 (not significant)
- R²: 0.011

**Conclusion**: No statistically significant or economically meaningful alpha. ESG characteristics appear already reflected in prices.

## Limitations

**Data**:
- Static ESG scores (single point-in-time)
- Survivorship bias (current S&P 100 constituents)
- Single provider (LSEG methodology)

**Statistical**:
- 7-year sample (relatively short for factor research)
- US large-cap only
- Monthly rebalancing (not institutional-realistic)

## Context

This project was developed to demonstrate:
- Factor construction from alternative data
- Statistical hypothesis testing
- Understanding of when factors don't work
- Realistic transaction cost modeling

The negative result is more valuable than finding a spurious effect - it reflects genuine market efficiency and distinguishes ESG as a values-alignment tool rather than an alpha source.

---

**Author**: Edouard Lavalard
Data from LSEG & Reifinitiv
