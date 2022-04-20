import yfinance as yf
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt

start = datetime(2004,1,1)
end = datetime.today()
SMI= yf.download("^SSMI", start, end)

SMI.to_csv("SMI_ALL.csv",index = False)

#Gleitender Mittelwert 50 / 200 2004 - 2022
SMI ["MA50"] = SMI["Close"].rolling(50).mean()
SMI ["MA200"] = SMI["Close"].rolling(200).mean()
SMI["Close"].plot(figsize = (20,9))
SMI["MA50"].plot()
SMI["MA200"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 200 Day Base SMI 2004-2022", fontsize = 30)

#Gleitender Mittelwert 50 / 200 2008 -2010

#get the data
start = datetime(2008,1,1)
end = datetime(2010,12,12)
SMI_08= yf.download("^SSMI", start, end)
SMI_08.to_csv("SMI_08.csv",index = False)
#plot
SMI_08 ["MA50"] = SMI_08["Close"].rolling(50).mean()
SMI_08 ["MA200"] = SMI_08["Close"].rolling(200).mean()
SMI_08["Close"].plot(figsize = (20,9))
SMI_08["MA50"].plot()
SMI_08["MA200"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 200 Day Base SMI 2008-2010", fontsize = 30)

# gibt den Mittelwert aus
(SMI_08["Close"].mean()


#Gleitender Mittelwert 50 / 200 2020 -2022

#get the data
start = datetime(2020,01,01)
end = datetime.today()
SMI_20= yf.download("^SSMI", start, end)

SMI_20.to_csv("SMI_08.csv",index = False)

#plot
SMI_20 ["MA50"] = SMI_20["Close"].rolling(50).mean()
SMI_20 ["MA200"] = SMI_20["Close"].rolling(200).mean()
SMI_20["Close"].plot(figsize = (20,9))
SMI_20["MA50"].plot()
SMI_20["MA200"].plot()
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.legend()
plt.title("Moving Average of a 50 and 200 Day Base SMI 2020-2022", fontsize = 30)

# gibt den Mittelwert aus
SMI_20["Close"].mean()


#Volatility

SMI_08["returns"] = (SMI_08["Close"]/SMI_08["Close"].shift(1)) -1
SMI_20["returns"] = (SMI_20["Close"]/SMI_20["Close"].shift(1)) -1
SMI_08["returns"].hist(bins = 100, label = "SMI 2008-2010", alpha = 0.5, figsize = (15,7))
SMI_20["returns"].hist(bins = 100, label = "SMI 2020-2022", alpha = 0.5)
plt.legend()
plt.title("Volatility SMI 2008-2010 and SMI 2020-2022", fontsize = 30)

# Comparison standard deviation SMI 2008-2010 and SMI 2022-2022
std_SMI_08= (SMI_08["Close"].std())
std_SMI_20= (SMI_20["Close"].std())
print(std_SMI_08)
print(std_SMI_20)




