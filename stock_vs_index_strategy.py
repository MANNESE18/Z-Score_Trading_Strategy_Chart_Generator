
# import required tools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf

# prompts user for inputs
stock_ticker = input("Enter Stock Ticker or Type 'quit' to exit:").upper()
index_ticker = input("Enter Index Ticker or Type 'quit' to exit:").upper()
span_1 = int(input("Enter span value for chart #1:"))
span_2 = int(input("Enter span value for chart #2:"))
z_score_1 = float(input("Enter z-score threshold for chart #1:"))
z_score_2 = float(input("Enter z-score threshold for chart #2:"))
start_year = int(input("Enter start year for study:"))
end_year = int(input("Enter end year for study:"))

data_start = start_year-1

data = yf.download([stock_ticker, index_ticker], start=f"{data_start}-01-01", auto_adjust=True)

# sets up data frames
stock_index_1 = data['Close'].copy()
stock_index_2 = data['Close'].copy()

stock_index_1 = stock_index_1.rename(columns={
    stock_ticker: 'ac_stock', 
    index_ticker: 'ac_index'
})

stock_index_2 = stock_index_2.rename(columns={
    stock_ticker: 'ac_stock', 
    index_ticker: 'ac_index'
})

# sets up instructions for creating the two trading strategy scenarios

# calculations for initial variables needed
stock_index_1['SI_Comp'] = stock_index_1['ac_stock']/stock_index_1['ac_index']
stock_index_1['IS_Comp'] = stock_index_1['ac_index']/stock_index_1['ac_stock']
stock_index_1['spread'] = np.maximum(stock_index_1['SI_Comp'], stock_index_1['IS_Comp'])
stock_index_1['spread_mean'] = stock_index_1['spread'].ewm(span=span_1).mean()
stock_index_1['spread_std'] = stock_index_1['spread'].ewm(span=span_1).std()
stock_index_1['z_score'] = (stock_index_1['spread'] - stock_index_1['spread_mean'])/(stock_index_1['spread_std'])

z_thresh_1 = z_score_1

# helps to define further conditions in code and how the "trades" will be signaled
stock_index_1['higher'] = np.maximum(stock_index_1['ac_stock'], stock_index_1['ac_index'])
stock_index_1['lower'] = np.minimum(stock_index_1['ac_stock'], stock_index_1['ac_index'])
stock_index_1['signal'] = np.where((stock_index_1['z_score'] >= z_thresh_1) | (stock_index_1['z_score'] <= -z_thresh_1), 1, 0)

# calculations for the weights of stock investment
stock_index_1['daily_return_for_higher'] = stock_index_1['higher'].pct_change()
stock_index_1['daily_return_for_lower'] = stock_index_1['lower'].pct_change()
stock_index_1['std_for_higher'] = stock_index_1['daily_return_for_higher'].ewm(span=span_1).std()
stock_index_1['std_for_lower'] = stock_index_1['daily_return_for_lower'].ewm(span=span_1).std()
stock_index_1['inv_std_sum'] = (1/stock_index_1['std_for_higher']) + (1/stock_index_1['std_for_lower'])
stock_index_1['weight_for_higher'] = (1/stock_index_1['std_for_higher']) / (stock_index_1['inv_std_sum'])
stock_index_1['weight_for_lower'] = (1/stock_index_1['std_for_lower']) / (stock_index_1['inv_std_sum'])

# if stock > index, then the action would always be short the stock
# if stock < index, then the action would always be long the stock
# only investment will be w/ the stock (instructions define whether to short or long the stock)
strategy_higher_1 = stock_index_1['higher'].pct_change()*stock_index_1['signal'].shift(1)*stock_index_1['weight_for_higher'].shift(1)*(-1)
strategy_lower_1 = stock_index_1['lower'].pct_change()*(stock_index_1['signal'].shift(1))*(stock_index_1['weight_for_lower'].shift(1))
stock_index_1['strategy_1'] = np.where(stock_index_1['higher'] == stock_index_1['ac_stock'], strategy_higher_1, strategy_lower_1)

# Sharpe ratio calculations for the strategy
strategy_daily_1 = stock_index_1['strategy_1'].fillna(0)
strategy_daily_mean_1 = strategy_daily_1.mean()
strategy_daily_std_1 = strategy_daily_1.std()
if strategy_daily_std_1 != 0:
    sharpe_1 = (strategy_daily_mean_1/strategy_daily_std_1)*np.sqrt(252)
else:
    sharpe_1 = 0
sharpe_text_1 = f'Annualized Sharpe: {sharpe_1: .2f}'

# calculates total return of investment to display
stock_index_1['return'] = ((1 + stock_index_1['strategy_1'].fillna(0)).cumprod()-1)
return_1 = stock_index_1['return'].iloc[-1]
return_text_1 = f'Total Return: {return_1:.2%}'

# calculates total number of trades to display
trade_count_1 = (stock_index_1['signal'].diff() == 1).sum()
trade_text_1 = f'Total Trades: {trade_count_1}'


# same instructions as above but it uses the second set of inputs from the user
stock_index_2['SI_Comp'] = stock_index_2['ac_stock']/stock_index_2['ac_index']
stock_index_2['IS_Comp'] = stock_index_2['ac_index']/stock_index_2['ac_stock']
stock_index_2['spread'] = np.maximum(stock_index_2['SI_Comp'], stock_index_2['IS_Comp'])
stock_index_2['spread_mean'] = stock_index_2['spread'].ewm(span=span_2).mean()
stock_index_2['spread_std'] = stock_index_2['spread'].ewm(span=span_2).std()
stock_index_2['z_score'] = (stock_index_2['spread'] - stock_index_2['spread_mean'])/(stock_index_2['spread_std'])

z_thresh_2 = z_score_2

stock_index_2['higher'] = np.maximum(stock_index_2['ac_stock'], stock_index_2['ac_index'])
stock_index_2['lower'] = np.minimum(stock_index_2['ac_stock'], stock_index_2['ac_index'])
stock_index_2['signal'] = np.where((stock_index_2['z_score'] >= z_thresh_2) | (stock_index_2['z_score'] <= -z_thresh_2), 1, 0)

stock_index_2['daily_return_for_higher'] = stock_index_2['higher'].pct_change()
stock_index_2['daily_return_for_lower'] = stock_index_2['lower'].pct_change()
stock_index_2['std_for_higher'] = stock_index_2['daily_return_for_higher'].ewm(span=span_2).std()
stock_index_2['std_for_lower'] = stock_index_2['daily_return_for_lower'].ewm(span=span_2).std()
stock_index_2['inv_std_sum'] = (1/stock_index_2['std_for_higher']) + (1/stock_index_2['std_for_lower'])
stock_index_2['weight_for_higher'] = (1/stock_index_2['std_for_higher']) / (stock_index_2['inv_std_sum'])
stock_index_2['weight_for_lower'] = (1/stock_index_2['std_for_lower']) / (stock_index_2['inv_std_sum'])

strategy_higher_2 = stock_index_2['higher'].pct_change()*stock_index_2['signal'].shift(1)*stock_index_2['weight_for_higher'].shift(1)*(-1)
strategy_lower_2 = stock_index_2['lower'].pct_change()*(stock_index_2['signal'].shift(1))*(stock_index_2['weight_for_lower'].shift(1))
stock_index_2['strategy_2'] = np.where(stock_index_2['higher'] == stock_index_2['ac_stock'], strategy_higher_2, strategy_lower_2)

strategy_daily_2 = stock_index_2['strategy_2'].fillna(0)
strategy_daily_mean_2 = strategy_daily_2.mean()
strategy_daily_std_2 = strategy_daily_2.std()
if strategy_daily_std_2 != 0:
    sharpe_2 = (strategy_daily_mean_2/strategy_daily_std_2)*np.sqrt(252)
else:
    sharpe_2 = 0
sharpe_text_2 = f'Annualized Sharpe: {sharpe_2: .2f}'

stock_index_2['return'] = ((1 + stock_index_2['strategy_2'].fillna(0)).cumprod()-1)
return_2 = stock_index_2['return'].iloc[-1]
return_text_2 = f'Total Return: {return_2:.2%}'

trade_count_2 = (stock_index_2['signal'].diff() == 1).sum()
trade_text_2 = f'Total Trades: {trade_count_2}'


# calculates certain stats to display for the strategy
stock_index_chart_1 = stock_index_1.truncate(before=f"{start_year}-01-01", after=f"{end_year}-12-31").copy()
stock_index_chart_2 = stock_index_2.truncate(before=f"{start_year}-01-01", after=f"{end_year}-12-31").copy()

cols_to_use = ['ac_stock', 'ac_index']
returns = stock_index_chart_1[cols_to_use].pct_change()
correlation = returns['ac_stock'].corr(returns['ac_index'])
r_squared = correlation ** 2
cov_matrix = returns.cov()
covariance = cov_matrix.loc['ac_stock', 'ac_index']
variance = returns['ac_index'].var()
beta = covariance / variance


# filter for desired dates after all calculations
stock_index_chart_1 = stock_index_1.truncate(before=f"{start_year}-01-01", after=f"{end_year}-12-31").copy()
stock_index_chart_2 = stock_index_2.truncate(before=f"{start_year}-01-01", after=f"{end_year}-12-31").copy()


# Buy and Hold returns for stock and index
stock_index_chart_1['stock_daily_return'] = stock_index_chart_1['ac_stock'].pct_change()
stock_index_chart_1['BandH_stock']= (1 + stock_index_chart_1['stock_daily_return'].fillna(0)).cumprod()-1
t_return_stock = stock_index_chart_1['BandH_stock'].iloc[-1]
t_stock_text_1 = f'Total Return: {t_return_stock:.2%}'

stock_index_chart_1['index_daily_return'] = stock_index_chart_1['ac_index'].pct_change()
stock_index_chart_1['BandH_index']= (1 + stock_index_chart_1['index_daily_return'].fillna(0)).cumprod()-1
t_return_index = stock_index_chart_1['BandH_index'].iloc[-1]
t_index_text_1 = f'Total Return: {t_return_index:.2%}'


# Sharpe ratio for stock and index
stock_daily = stock_index_chart_1['stock_daily_return']
stock_daily_mean = stock_daily.mean()
stock_daily_std = stock_daily.std()
if stock_daily_std != 0:
    stock_sharpe = (stock_daily_mean/stock_daily_std)*np.sqrt(252)
else:
    stock_sharpe = 0
stock_sharpe_text = f'Annualized Sharpe: {stock_sharpe: .2f}'

index_daily = stock_index_chart_1['index_daily_return']
index_daily_mean = index_daily.mean()
index_daily_std = index_daily.std()
if index_daily_std != 0:
    index_sharpe = (index_daily_mean/index_daily_std)*np.sqrt(252)
else:
    index_sharpe = 0
index_sharpe_text = f'Annualized Sharpe: {index_sharpe: .2f}'


# stock/index sharpe+return setup in chart
stock_text_both = f'{stock_ticker}: {stock_sharpe_text} | {t_stock_text_1}'
index_text_both = f'{index_ticker}: {index_sharpe_text} | {t_index_text_1}'


# chart setup
fig, ((a,b),(c,d)) = plt.subplots(2,2, figsize=(12,10), sharex=True)

fig.suptitle(f'Z-Score Strategy: {stock_ticker} | Correlation: {correlation:.2f} | R-Squared: {r_squared:.2f} | Beta: {beta:.2f}', fontsize=16)

# z-score chart for strategy #1
stock_index_chart_1.plot(ax=a, y='z_score')
a.set_title(f'Z-Score Signal: {z_score_1}-day Tactical Window (Span: {span_1})')
a.set_xlabel('Year')
a.set_ylabel('Z-Score')
a.axhline(z_score_1, color='black', linestyle='--', linewidth=1.5, label=f'Sell/Short Threshold ({z_score_1}σ)')
a.axhline(-z_score_1, color='black', linestyle='--', linewidth=1.5, label=f'Sell/Short Threshold (-{z_score_1}σ)')
a.fill_between(stock_index_chart_1.index, z_score_1, stock_index_chart_1['z_score'], 
               where=(stock_index_chart_1['z_score'].values >= z_score_1), 
               color='gray', alpha=0.4, label='Overbought')
a.fill_between(stock_index_chart_1.index, -z_score_1, stock_index_chart_1['z_score'], 
               where=(stock_index_chart_1['z_score'].values <= -z_score_1), 
               color='gray', alpha=0.4, label='Oversold')
a.axhline(0, color='black', linewidth=1)
a.grid(True, alpha=.5)

stock_index_chart_1.plot(ax=c, y='return', label = 'Strategy')
stock_index_chart_1.plot(ax=c, y='BandH_stock', color='orange', label=f'{stock_ticker}')
stock_index_chart_1.plot(ax=c, y='BandH_index',color='blue', label = f'{index_ticker}')

# investment returns chart for strategy #1
c.set_title(f'Performance: Aggressive Mean Reversion (Z={z_score_1})')
c.set_xlabel('Year')
c.set_ylabel('Total Return (%)')
c.text(0.02, 0.95, sharpe_text_1, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.text(0.02, 0.88, return_text_1, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.text(0.02, 0.81, trade_text_1, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.text(0.02, 0.12, stock_text_both, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.text(0.02, 0.05, index_text_both, transform = c.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
c.fill_between(stock_index_chart_1.index,c.get_ylim()[0], c.get_ylim()[1], where = (stock_index_chart_1['signal'] !=0), color='gray', alpha=0.15, label='In Market')
c.axhline(0, color='black', linewidth=1, alpha=0.5)
c.grid(True, alpha=.5)


# z-score chart for strategy #2
stock_index_chart_2.plot(ax=b, y='z_score')
b.set_title(f'Z-Score Signal: {z_score_2}-day Tactical Window (Span: {span_2})')
b.set_xlabel('Year')
b.set_ylabel('Z-Score')
b.axhline(z_score_2, color='black', linestyle='--', linewidth=1.5, label=f'Sell/Short Threshold ({z_score_2}σ)')
b.axhline(-z_score_2, color='black', linestyle='--', linewidth=1.5, label=f'Sell/Short Threshold (-{z_score_2}σ)')
b.fill_between(stock_index_chart_2.index, z_score_2, stock_index_chart_2['z_score'], 
               where=(stock_index_chart_2['z_score'].values >= z_score_2), 
               color='gray', alpha=0.4, label='Overbought')
b.fill_between(stock_index_chart_2.index, -z_score_2, stock_index_chart_2['z_score'], 
               where=(stock_index_chart_2['z_score'].values <= -z_score_2), 
               color='gray', alpha=0.4, label='Oversold')
b.axhline(0, color='black', linewidth=1)
b.grid(True, alpha=.5)

# investment returns chart for strategy #2
stock_index_chart_2.plot(ax=d, y='return', label = 'Strategy')
stock_index_chart_1.plot(ax=d, y='BandH_stock', color='orange', label=f'{stock_ticker}')
stock_index_chart_1.plot(ax=d, y='BandH_index',color='blue', label = f'{index_ticker}')
d.set_title(f'Performance: Aggressive Mean Reversion (Z={z_score_2})')
d.set_xlabel('Year')
d.set_ylabel('Total Return (%)')
d.text(0.02, 0.95, sharpe_text_2, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.text(0.02, 0.88, return_text_2, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.text(0.02, 0.81, trade_text_2, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.text(0.02, 0.12, stock_text_both, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.text(0.02, 0.05, index_text_both, transform = d.transAxes, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
d.fill_between(stock_index_chart_2.index,d.get_ylim()[0], d.get_ylim()[1], where = (stock_index_chart_2['signal'] !=0), color='gray', alpha=0.15, label='In Market')
d.axhline(0, color='black', linewidth=1, alpha=0.5)
d.grid(True, alpha=.5)

plt.tight_layout()
plt.show()
    


          













          
          
   
