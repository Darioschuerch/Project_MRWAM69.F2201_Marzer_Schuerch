import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix

#prices
start = datetime(2018,12,31)
end = datetime.today()
tsla = yf.download("tsla", start, end)

tsla["Close"].plot(label = "TSLA", figsize = (15,7))
plt.title("Stock Prices of Tesla", fontsize = 20)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.show()

#Gleitender Mittelwert 50 / 200
tsla ["MA50"] = tsla["Close"].rolling(50).mean()
tsla ["MA200"] = tsla["Close"].rolling(200).mean()
tsla["Close"].plot(figsize = (15,7))
tsla["MA50"].plot()
tsla["MA50"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 200 Day Base", fontsize = 20)

#Volatility
tsla['returns'] = (tsla['Close']/tsla['Close'].shift(1)) -1
tsla['returns'].hist(bins = 100, label = 'TSLA', alpha = 0.5, figsize = (15,7))
plt.legend()
plt.title("Volatility", fontsize = 20)