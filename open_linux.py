import pandas as pd
import numpy as np
import ctypes
from ctypes import POINTER, c_int, c_double

# 加载OpenBlas库
openblas = ctypes.cdll.LoadLibrary("libopenblas.so")

# 示例数据
data = {
    'date': [20230101, 20230102, 20230103, 20230104, 20230105, 20230106, 20230107, 20230108, 20230109, 20230110],
    'ticker': ['AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL', 'AAPL'],
    'return': [0.01, 0.02, -0.01, 0.03, 0.04, 0.01, 0.02, -0.01, 0.03, 0.04],
    'weight': [0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]
}
df = pd.DataFrame(data)

def Market_return(df, date):
    """Calculate the weighted average return for a given date"""
    subset = df[df['date'] == date]
    weighted_returns = subset['return'] * subset['weight']
    market_return = weighted_returns.sum() / subset['weight'].sum()
    return market_return

def Beta_openblas(df, date, ticker):
    """Calculate the beta of the given ticker on the given date using OpenBlas"""
    # Filter data for the specific ticker and dates before the given date
    subset = df[(df['ticker'] == ticker) & (df['date'] < date)]
    
    # Sort by date and take the last 128 entries
    subset = subset.sort_values(by='date').tail(128)
    
    # If there are less than 128 entries, start from the earliest date
    if len(subset) < 128:
        subset = df[df['ticker'] == ticker].sort_values(by='date')
    
    # Calculate market returns for the same dates
    market_returns = np.array([Market_return(df, d) for d in subset['date']])
    
    # Prepare the data for linear regression
    stock_returns = subset['return'].values
    n = len(market_returns)
    
    # Define the OpenBlas functions we need
    openblas.cblas_dgemv.argtypes = [
        c_int, c_int, c_int, c_int, c_double, 
        POINTER(c_double), c_int, POINTER(c_double), c_int, 
        c_double, POINTER(c_double), c_int
    ]
    
    openblas.cblas_dgemm.argtypes = [
        c_int, c_int, c_int, c_int, c_int, c_int, c_double, 
        POINTER(c_double), c_int, POINTER(c_double), c_int, 
        c_double, POINTER(c_double), c_int
    ]
    
    openblas.LAPACKE_dgesv.argtypes = [
        c_int, c_int, c_int, POINTER(c_double), c_int, POINTER(c_int), 
        POINTER(c_double), c_int
    ]

    X = np.vstack([market_returns, np.ones(n)]).T
    X_flat = X.flatten().astype(np.float64)
    XTX = np.empty((2, 2), dtype=np.float64)
    XTY = np.empty(2, dtype=np.float64)

    openblas.cblas_dgemm(
        101, 111, 111, 2, 2, n, 1.0, 
        X_flat.ctypes.data_as(POINTER(c_double)), 2, 
        X_flat.ctypes.data_as(POINTER(c_double)), 2, 
        0.0, XTX.ctypes.data_as(POINTER(c_double)), 2
    )

    openblas.cblas_dgemv(
        101, 111, n, 2, 1.0, 
        X_flat.ctypes.data_as(POINTER(c_double)), 2, 
        stock_returns.ctypes.data_as(POINTER(c_double)), 1, 
        0.0, XTY.ctypes.data_as(POINTER(c_double)), 1
    )

    # Create arrays to hold solution
    beta = np.empty(2, dtype=np.float64)
    ipiv = np.empty(2, dtype=np.int32)
    
    # Solve the system
    result = openblas.LAPACKE_dgesv(
        101, 2, 1, XTX.ctypes.data_as(POINTER(c_double)), 2, 
        ipiv.ctypes.data_as(POINTER(c_int)), XTY.ctypes.data_as(POINTER(c_double)), 2
    )
    
    if result == 0:
        beta[:] = XTY[:]
    else:
        raise ValueError("Linear system could not be solved")
    
    return beta