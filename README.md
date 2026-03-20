# Stock vs. Index Z-Score Mean Reversion Strategy


This project implements a quantitative pair-trading strategy that evaluates the relative strength between a specific equity and a benchmark index. By utilizing a Z-Score calculation derived from the price ratio of the two assets, the strategy identifies statistical outliers (overbought or oversold conditions) to execute mean-reversion trades. The tool allows for side-by-side backtesting of two different tactical windows and sensitivity thresholds, providing a visual and statistical comparison of how different parameters affect performance metrics like the Sharpe Ratio and Total Return.


## Features

* **Dynamic Data Retrieval**: Integrates with the yfinance API to fetch real-time and historical price data for any valid stock and index ticker.

* **Comparative Backtesting**: Supports dual-charting to compare two distinct strategy configurations (different spans and Z-score thresholds) simultaneously.

* **Volatility-Adjusted Weighting**: Implements an inverse-volatility weighting scheme to balance the position sizes between the "higher" and "lower" priced assets in the pair.

* **Statistical Benchmarking**: Automatically calculates and displays Correlation, R-Squared, and Beta between the stock and the index to provide context for the strategy's viability.

* **Performance Visualization**: Generates professional-grade Matplotlib dashboards featuring Z-score signal tracking, cumulative return curves, and shaded "In Market" periods.


## Built With


* **Python**: The core programming language used for logic and execution.

* **Pandas**: For high-performance data manipulation, time-series alignment, and EWM (Exponential Weighted Moving) calculations.

* **NumPy**: For vectorized mathematical operations and efficient conditional signal generation.

* **Matplotlib**: For constructing the multi-panel visual dashboard and performance charts.

* **YFinance**: To stream historical market data directly into the analytical pipeline.


## Key Achievements in Code


* **Automated Signal Logic**: Developed a robust np.where logic gate that identifies entry and exit points based on user-defined standard deviation (σ) thresholds, handling both long and short regimes seamlessly.

* **Vectorized Backtesting**: Built a cumulative return engine using .pct_change() and .cumprod(), allowing for rapid performance calculation across years of data without the need for slow iterative loops.

* **Advanced Statistical Indicators**: Successfully implemented automated calculations for the Sharpe Ratio (annualized by  
252), Beta, and R-Squared, providing institutional-grade risk metrics for the strategy.

Adaptive Charting: Designed a flexible Matplotlib subplot architecture that shares X-axis data (dates) while independently rendering tactical windows and return data for two different strategy variants.
