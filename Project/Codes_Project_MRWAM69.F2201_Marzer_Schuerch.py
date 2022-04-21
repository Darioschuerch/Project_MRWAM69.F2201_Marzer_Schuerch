import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

start = datetime(2004,1,1)
end = datetime.today()
SMI= yf.download("^SSMI", start, end)

SMI.to_csv("SMI_ALL.csv")


#Gleitender Mittelwert 50 / 200 2004 - 2022
SMI ["MA50"] = SMI["Close"].rolling(50).mean()
SMI ["MA200"] = SMI["Close"].rolling(200).mean()
SMI["Close"].plot(figsize = (10,5))
SMI["MA50"].plot()
SMI["MA200"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 200 Day Base SMI 2004-2022", fontsize = 30)


#get the data
start = datetime(2008,1,1)
end = datetime(2010,12,12)
SMI_08= yf.download("^SSMI", start, end)
SMI_08.to_csv("SMI_08.csv")

#plot
SMI_08 ["MA50"] = SMI_08["Adj Close"].rolling(50).mean()
SMI_08 ["MA100"] = SMI_08["Adj Close"].rolling(100).mean()
SMI_08["Adj Close"].plot(figsize = (20,9))
SMI_08["MA50"].plot()
SMI_08["MA100"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 100 Day Base SMI 2008-2010", fontsize = 30)

#Gleitender Mittelwert 50 / 200 2020 -2022

#get the data
start = datetime(2019,1,1)
end = datetime(2021,12,31)
SMI_19= yf.download("^SSMI", start, end)

SMI_19.to_csv("SMI_19.csv")

#plot
SMI_19 ["MA50"] = SMI_19["Adj Close"].rolling(50).mean()
SMI_19 ["MA100"] = SMI_19["Adj Close"].rolling(100).mean()
SMI_19["Close"].plot(figsize = (20,9))
SMI_19["MA50"].plot()
SMI_19["MA100"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 100 Day Base SMI 2019-2021", fontsize = 30)

# gibt den Mittelwert SMI 08 / 19 aus

mean_SMI_08 = SMI_08["Adj Close"].mean()
mean_SMI_19 = SMI_19["Adj Close"].mean()
print(mean_SMI_08)
print(mean_SMI_19)


#Volatility

SMI_08["returns"] = (SMI_08["Adj Close"]/SMI_08["Adj Close"].shift(1)) -1
SMI_19["returns"] = (SMI_19["Adj Close"]/SMI_19["Adj Close"].shift(1)) -1
SMI_08["returns"].hist(bins = 100, label = "SMI 2008-2010", alpha = 0.5, figsize = (15,7))
SMI_19["returns"].hist(bins = 100, label = "SMI 2019-2021", alpha = 0.5)
plt.legend()
plt.title("Volatility SMI 2008-2010 and SMI 2019-2021", fontsize = 30)

# Comparison standard deviation SMI 2008-2010 and SMI 2019-2022

std_SMI_08= SMI_08["Adj Close"].std()
std_SMI_19= SMI_19["Adj Close"].std()
print(std_SMI_08)
print(std_SMI_19)


#Regression
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#indexiert mit DAX








