import pandas as pd
import numpy as np

data = {
    'date': [20230101, 20230101, 20230102, 20230102, 20230103],
    'ticker': ['AAPL', 'IBM', 'AAPL', 'IBM', 'AAPL'],
    'return': [0.01, 0.02, 0.015, 0.025, 0.03],
    'weight': [0.6, 0.4, 0.6, 0.4, 0.6]
}

df = pd.DataFrame(data)


#Method 1 for Market Return
def market_return(date, df):
    daily_data = df[df['date'] == date]
    weighted_avg = np.average(daily_data['return'], weights=daily_data['weight'])
    return weighted_avg

# Method 2 : using Pivot Table
#print(market_return(20230101, df))

def market_return_pivot(date, df):
    pivot = df.pivot_table(values='return', index='date', columns='ticker', aggfunc='sum', fill_value=0)
    weights = df.pivot_table(values='weight', index='date', columns='ticker', aggfunc='sum', fill_value=0)
    if date in pivot.index:
        weighted_avg = np.average(pivot.loc[date], weights=weights.loc[date])
        return weighted_avg
    return None

# Example usage
#print(market_return_pivot(20230101, df))


def beta_manual(date, ticker, df):
    # 过滤ticker数据
    ticker_data = df[df['ticker'] == ticker]
    
    # 按日期排序
    ticker_data = ticker_data.sort_values(by='date')
    
    # 找到当前日期的索引
    current_index = ticker_data[ticker_data['date'] == date].index[0]
    
    # 确定滚动窗口的起始索引
    start_index = max(0, current_index - 128)
    
    # 滚动窗口的数据
    window_data = ticker_data[start_index:current_index]
    
    if len(window_data) < 2:
        return None  # 数据点不足
    
    # 计算窗口内每个日期的市场回报
    window_data['market_return'] = window_data['date'].apply(lambda d: market_return(d, df))
    
    # 获取市场回报和个股回报
    X = window_data['market_return'].values
    y = window_data['return'].values
    print(y.shape)
    print(X.shape)
    
    # 计算均值
    X_mean = np.mean(X)
    y_mean = np.mean(y)
    
    # 计算回归系数
    numerator = np.sum((X - X_mean) * (y - y_mean))
    denominator = np.sum((X - X_mean) ** 2)
    
    if denominator == 0:
        return None  # 避免除以零
    
    beta_value = numerator / denominator
    
    return beta_value

# 示例使用
print(beta_manual(20230103, 'AAPL', df))
# 示例使用
#print(beta(20230103, 'AAPL', df))