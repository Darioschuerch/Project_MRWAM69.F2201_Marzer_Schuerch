#!pip install yfinance
#!pip install fix-yahoo-finance
#!pip install yfinance
#!pip install datetime
#!pip install pandas_datareader
import numpy as np
import matplotlib.pyplot as plt
import scipy as scipy
import yfinance as yf
import pandas as pd
import datetime as dt
from datetime import datetime
from pandas_datareader import data as pdr
from scipy.stats import norm

#Get the data
start = datetime(2004,1,1)
end = datetime.today()
SMI= yf.download("^SSMI", start, end)

SMI.to_csv("SMI_ALL.csv")


#Gleitender Mittelwert 50 / 200 2004 - 2022
SMI["MA50"] = SMI["Adj Close"].rolling(50).mean()
SMI["MA200"] = SMI["Adj Close"].rolling(200).mean()
SMI["Adj Close"].plot(figsize = (15,4))
SMI["MA50"].plot()
SMI["MA200"].plot()
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()
plt.ylabel(" CHF", fontsize =15)
plt.xlabel("Date", fontsize =15)
plt.title("Moving Averages SMI 2004-2022", fontsize = 25)


#Gleitender Mittelwert 50 / 200 2008-2010
#get the data
start = datetime(2008,1,1)
end = datetime(2010,12,12)
SMI_08= yf.download("^SSMI", start, end)
SMI_08.to_csv("SMI_08.csv")

#plot
SMI_08 ["MA50"] = SMI_08["Adj Close"].rolling(50).mean()
SMI_08 ["MA100"] = SMI_08["Adj Close"].rolling(100).mean()
SMI_08["Adj Close"].plot(figsize = (15,4))
SMI_08["MA50"].plot()
SMI_08["MA100"].plot()
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()
plt.ylabel(" CHF", fontsize =15)
plt.xlabel("Date", fontsize =15)
plt.title("Moving Averages SMI 2008-2010", fontsize = 25)

#Gleitender Mittelwert 50 / 200 2019 -2022

#get the data
start = datetime(2019,1,1)
end = datetime(2021,12,31)
SMI_19= yf.download("^SSMI", start, end)
SMI_19.to_csv("SMI_19.csv")

#plot
SMI_19["MA50"] = SMI_19["Adj Close"].rolling(50).mean()
SMI_19["MA100"] = SMI_19["Adj Close"].rolling(100).mean()
SMI_19["Adj Close"].plot(figsize = (15,4))
SMI_19["MA50"].plot()
SMI_19["MA100"].plot()
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()
plt.ylabel(" CHF", fontsize =15)
plt.xlabel("Date", fontsize =15)
plt.title("SMI 2019-2021: Moving Average of a 50 and 100 Day Base", fontsize = 25)


# gibt den Mittelwert SMI 08 / 19 aus

mean_SMI_08 = SMI_08["Adj Close"].mean()
mean_SMI_19 = SMI_19["Adj Close"].mean()
print(mean_SMI_08)
print(mean_SMI_19)

## Volatility Clusterting SMI 2004-2022
#Calculate Pct Change & Log Return (steady return)
df_04 = pd.read_csv("SMI_ALL.csv",sep = ",", index_col=0)
df_04["Pct Change"]= df_04["Adj Close"].pct_change()
df_04["Log Return"]= np.log(df_04["Adj Close"]/df_04["Adj Close"].shift(1))
df_04 =df_04[1:] #Remove the first line (NaN)
df_04.head()
#ploting volatility clustering
df_04["Log Return"].plot(figsize = (20,7))
plt.title("Volatility Clustering of SMI 2004-2022",  fontsize = 30)
plt.xlabel("Date",  fontsize = 25)


# Comparison Volatility SMI 2008-2010 and SMI 2019-2021

#data 08: calculate pct change & log return

df_08 = pd.read_csv("SMI_08.csv",sep = ",", index_col=0)
df_08["Pct Change"]= df_08["Adj Close"].pct_change()
df_08["Log Return"]= np.log(df_08["Adj Close"]/df_08["Adj Close"].shift(1))
df_08 =df_08[1:] #Remove the first line (NaN)
df_08.head()

#data 19: calculate pct change & log return
df_19 = pd.read_csv("SMI_19.csv",sep = ",", index_col=0)
df_19["Pct Change"]= df_19["Adj Close"].pct_change()
df_19["Log Return"]= np.log(df_19["Adj Close"]/df_19["Adj Close"].shift(1))
df_19 =df_19[1:] #Remove the first line (NaN)
df_19.head()
# Note: pct Change and Log return calculate nearly the same values

# Volatilität SMI über 3 Börsenjahre in den jeweiligen Krisen (252 *3)
std_SMI_08= df_08["Log Return"].std()
std_SMI_19= df_19["Log Return"].std()
std_SMI_08_3Y = std_SMI_08 * np.sqrt(756) * 100
std_SMI_19_3Y = std_SMI_19 * np.sqrt(756) * 100
print(std_SMI_08_3Y)
print(std_SMI_19_3Y)


#plot volatility histogram
df_08["Log Return"].hist(bins = 100, label = "Log Returns SMI 2008-2010", alpha = 0.5, figsize = (15,7))
df_19["Log Return"].hist(bins = 100, label = "Log Returns 2019-2021", alpha = 0.5)
plt.legend()
plt.ylabel("Frequency", fontsize =15)
plt.xlabel("Log Returns", fontsize =15)
plt.title("Volatility SMI 2008-2010 and SMI 2019-2021", fontsize = 30)

#Boxplot 2008-2010
box_08 = df_08["Log Return"]
plt.figure(figsize = (10,7))
plt.title("Boxplot Log Returns SMI  2008-2010", fontsize = (25))
ax = box_08.plot.box()

#Boxplot 2019-2021
box_19 = df_19["Log Return"]
plt.figure(figsize = (10,7))
plt.title("Boxplot Log Returns SMI  2019-2021", fontsize = (25))
ax = box_19.plot.box()

#Übersicht portfolio
start = datetime(2004,1,1)
end = datetime(2021,12,31)
df_NOVN= yf.download("NOVN.SW", start, end)
Base_100 = df_NOVN.iat[0,4]
df_NOVN["Base 100"] = (df_NOVN["Adj Close"]/Base_100 *100)
df_NOVN = df_NOVN[1:]
df_NOVN.head() #Overview Novartis

df_UBSG= yf.download("UBSG.SW", start, end)
Base_100 = df_UBSG.iat[0,4]
df_UBSG["Base 100"] = (df_UBSG["Adj Close"]/Base_100 *100)
df_UBSG = df_UBSG[1:]

df_NESN= yf.download("NESN.SW", start, end)
Base_100 = df_NESN.iat[0,4]
df_NESN["Base 100"] = (df_NESN["Adj Close"]/Base_100 *100)
df_NESN = df_NESN[1:]

df_CSGN= yf.download("CSGN.SW", start, end)
Base_100 = df_CSGN.iat[0,4]
df_CSGN["Base 100"] = (df_CSGN["Adj Close"]/Base_100 *100)
df_CSGN = df_CSGN[1:]

df_ABBN= yf.download("ABBN.SW", start, end)
Base_100 = df_ABBN.iat[0,4]
df_ABBN["Base 100"] = (df_ABBN["Adj Close"]/Base_100 *100)
df_ABBN = df_ABBN[1:]

start = datetime(2004,1,1)
end = datetime.today()
plt.figure(figsize = (15,7))
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)
plt.ylabel("Indexpoints", fontsize =15)
plt.xlabel("Date", fontsize =15)
plt.plot(df_NOVN["Base 100"], label ="NOVN.SW")
plt.plot(df_UBSG["Base 100"], label ="UBSG.SW")
plt.plot(df_CSGN["Base 100"], label ="CSGN.SW")
plt.plot(df_NESN["Base 100"], label ="NESN.SW")
plt.plot(df_ABBN["Base 100"], label ="ABBN.SW")
plt.title("Adjusted closing prices of the SMI sample Portfolio", fontsize = 25)
plt.legend()


#Max Portfolio loss  5 Titel two months during Financial Crisis
portfolio_08 = ["NOVN.SW", "UBSG.SW", "NESN.SW", "CSGN.SW", "ABBN.SW"]
weights_08 = np.array([.2, .2, .2, .2, .2])
investment_08 = 1000000
start_08 = datetime(2008,10,1)
end_08 = datetime(2008,12,1)
df_08 = pdr.get_data_yahoo(portfolio_08, start_08, end_08) ["Adj Close"]
returns_08 = df_08.pct_change()

# Generate Var-Cov matrix
cov_matrix_08 = returns_08.cov()
#print(cov_matrix)

#calculate mean returns of the stocks
mean_returns_08 = returns_08.mean()

#calculate mean returns for the portfolio and normalize against investments weights
mean_portfolio_08 = mean_returns_08.dot(weights_08)

#Standard deviation of the portfolio
std_portfolio_08 = np.sqrt(weights_08.T.dot(cov_matrix_08).dot(weights_08))

#Mean of investment
mean_investment_08 = (1+mean_portfolio_08) * investment_08

#Standard deviation of investment
std_investment_08 = investment_08 * std_portfolio_08

#confidence interval (95%)
alpha_08 = 0.05
var_cutoff_08 = norm.ppf(alpha_08, mean_investment_08, std_investment_08) #normal cumulative distribution
Var_08 = investment_08 - var_cutoff_08
#print(Var_08)


#Calculate VaR over 2 months
import matplotlib.pyplot as plt
Var_array_08= []
days_08 = int(60)
for x in range(1, days_08+1):
    Var_array_08.append(np.round(Var_08 * np.sqrt(x), 2))
    # acitvate the following code to see VaR over two months
    #print(str(x) + " day VaR @ 95% confidence: " + str(np.round(Var_08 * np.sqrt(x), 2)))

plt.figure(figsize=(15,7))
plt.xlabel("Days", fontsize = 15)
plt.ylabel("Max. portfolio loss (CHF)", fontsize = 15)
plt.xlim(0,70)
plt.ylim(0,800_000)
plt.title("Maximum portfolio loss over two months", fontsize = 25)
plt.plot(Var_array_08, "b")
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)

#Calculating VaR 2008-2010
returns_08 = returns_08.fillna(0.0)
portfolio_returns_08 = returns_08.iloc[-days_08:].dot(weights_08)

VaR_08 = np.percentile(portfolio_returns_08, 100 * (alpha_08)) * investment_08
print(VaR_08)  # max loss with a conf level of 95% is 32.3k


#plot
portfolio_returns_08 = returns_08.fillna(0.0).iloc[-days_08:].dot(weights_08)

portfolio_VaR_08 = VaR_08
portfolio_VaR_return_08 = portfolio_VaR_08 / investment_08

plt.figure(figsize=(15,7))
plt.hist(portfolio_returns_08, bins= 50)
plt.axvline(portfolio_VaR_return_08, color="r", linestyle="solid")
plt.legend(["VaR for alpha = 5%", "Historical Returns SMI Portfolio" ])
plt.title("Maximum portfolio loss in the period 2008-2010", fontsize = 25)
plt.xlabel("Return", fontsize = 20)
plt.ylabel("Observation Frequency", fontsize = 20)
plt.grid(which="major", color='k', linestyle='-.', linewidth=0.3)


#Checking distributions of equities against normal distribution
#Wie im Abschnitt über die Berechnung erwähnt, gehen wir bei der Berechnung des VaR davon aus, dass die Renditen der Aktien in unserem Portfolio normal verteilt sind. Natürlich können wir das für die Zukunft nicht vorhersagen, aber wir können zumindest prüfen, wie die historischen Renditen verteilt waren, um zu beurteilen, ob der VaR für unser Portfolio geeignet ist
import matplotlib.pyplot as plt
import scipy as scipy
returns_08["NOVN.SW"].hist(bins = 100, label = "NOVN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_08 - 3* std_portfolio_08, mean_portfolio_08 +3 *std_portfolio_08,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_08, std_portfolio_08), "r")
plt.title("Novartis returns 2008-2010 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_08["UBSG.SW"].hist(bins = 100, label = "UBSG.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_08 - 3* std_portfolio_08, mean_portfolio_08 +3 *std_portfolio_08,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_08, std_portfolio_08), "r")
plt.title("UBS returns 2008-2010 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_08["NESN.SW"].hist(bins = 100, label = "NESN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_08 - 3* std_portfolio_08, mean_portfolio_08 +3 *std_portfolio_08,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_08, std_portfolio_08), "r")
plt.title("Nestle returns 2008-2010 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_08["CSGN.SW"].hist(bins = 100, label = "CSGN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_08 - 3* std_portfolio_08, mean_portfolio_08 +3 *std_portfolio_08,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_08, std_portfolio_08), "r")
plt.title("Credit Suisse returns 2008-2010 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_08["ABBN.SW"].hist(bins = 100, label = "ABBN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_08 - 3* std_portfolio_08, mean_portfolio_08 +3 *std_portfolio_08,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_08, std_portfolio_08), "r")
plt.title("ABB returns 2008-2010 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

#VaR 5  Titel 2019-2021  during COVID
portfolio_19 = ["NOVN.SW", "UBSG.SW", "NESN.SW", "CSGN.SW", "ABBN.SW"]
weights_19 = np.array([.2, .2, .2, .2, .2])
investment_19 = 1000000
start_19 = datetime(2020,3,1)
end_19 = datetime(2020,5,1)
df_19 = pdr.get_data_yahoo(portfolio_19, start_19, end_19) ["Adj Close"]
returns_19 = df_19.pct_change()
#print(returns)

# Generate Var-Cov matrix
cov_matrix_19 = returns_19.cov()
#print(cov_matrix)

#calculate mean returns of the stocks
mean_returns_19 = returns_19.mean()

#calculate mean returns for the portfolio and normalize against investments weights
mean_portfolio_19 = mean_returns_19.dot(weights_19)

#Standard deviation of the portfolio
std_portfolio_19 = np.sqrt(weights_19.T.dot(cov_matrix_19).dot(weights_19))

#Mean of investment
mean_investment_19 = (1+mean_portfolio_19) * investment_19

#Standard deviation of investment
std_investment_19 = investment_19 * std_portfolio_19

#Confidenceinterval (95%)
alpha_19 = 0.05
var_cutoff_19 = norm.ppf(alpha_19, mean_investment_19, std_investment_19) #normal cumulative distribution
Var_19 = investment_19 - var_cutoff_19
#print(Var_19)

#Calculate VaR over 2 months
Var_array_19= []
days_19 = int(60)
for x in range(1, days_19+1):
    Var_array_19.append(np.round(Var_19 * np.sqrt(x), 2))
    # acitvate code to see VaR over 2 months
    #print(str(x) + " day VaR @ 95% confidence: " + str(np.round(Var_19 * np.sqrt(x), 2)))


plt.figure(figsize=(15,7))
plt.xlabel("Days", fontsize = 15)
plt.ylabel("Max. portfolio loss (CHF)", fontsize = 15)
plt.xlim(0,70)
plt.ylim(0,700_000)
plt.title("Maximum portfolio loss over two months", fontsize = 25)
plt.plot(Var_array_19, "b")
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)

#Calculating VaR 2019-2021
returns_19 = returns_19.fillna(0.0)
portfolio_returns_19 = returns_19.iloc[-days_19:].dot(weights_19)
VaR_19 = np.percentile(portfolio_returns_19, 100 * (alpha_19)) * investment_19
print(VaR_19)  # max loss with a conf level of 95% is 18.4k

portfolio_returns_19_ = returns_19.fillna(0.0).iloc[-days_19:].dot(weights_19)

portfolio_VaR_19 = VaR_19
portfolio_VaR_return_19 = portfolio_VaR_19 / investment_19

plt.figure(figsize=(15,7))
plt.hist(portfolio_returns_19, bins= 50)
plt.axvline(portfolio_VaR_return_19, color="r", linestyle="solid")
plt.legend(["VaR for alpha = 5%", "Historical Returns SMI Portfolio" ])
plt.title("Maximum portfolio loss in the period 2019-2021", fontsize = 25)
plt.xlabel("Returns", fontsize = 20)
plt.ylabel("Observation Frequency", fontsize = 20)
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)

#Checking distributions of equities against normal distribution

returns_19["NOVN.SW"].hist(bins = 100, label = "NOVN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_19 - 3* std_portfolio_19, mean_portfolio_19 +3 *std_portfolio_19,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_19, std_portfolio_19), "r")
plt.title("Novartis returns 2019-2021 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_19["UBSG.SW"].hist(bins = 100, label = "UBSG.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_19 - 3* std_portfolio_19, mean_portfolio_19 +3 *std_portfolio_19,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_19, std_portfolio_19), "r")
plt.title("UBS returns 2019-2021 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_19["NESN.SW"].hist(bins = 100, label = "NESN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_19 - 3* std_portfolio_19, mean_portfolio_19 +3 *std_portfolio_19,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_19, std_portfolio_19), "r")
plt.title("Nestle returns 2019-2021 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_19["CSGN.SW"].hist(bins = 100, label = "CSGN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_19 - 3* std_portfolio_19, mean_portfolio_19 +3 *std_portfolio_19,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_19, std_portfolio_19), "r")
plt.title("Credit Suisse returns 2019-2021 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

returns_19["ABBN.SW"].hist(bins = 100, label = "ABBN.SW", alpha = 0.5, figsize = (10,4))
x = np.linspace(mean_portfolio_19 - 3* std_portfolio_19, mean_portfolio_19 +3 *std_portfolio_19,100)
plt.plot(x, scipy.stats.norm.pdf(x, mean_portfolio_19, std_portfolio_19), "r")
plt.title("ABB returns 2019-2021 vs. normal distribution", fontsize = (20))
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.3)
plt.legend()

#Sharp ratio Portfolio 2008-2010
returns_08 = np.sum(np.multiply(mean_returns_08, weights_08)) * 504
std_08 = std_portfolio_08
risk_free_return_08 = int(0.018) # assumption
sharpe_ratio_08= (mean_portfolio_08 - risk_free_return_08) / std_08
print(sharpe_ratio_08)

#Sharp ratio Portfolio 2019-2021
returns_19 = np.sum(np.multiply(mean_returns_08, weights_08)) * 504
std_19 = std_portfolio_19
risk_free_return_19 = 0 # assumption
sharpe_ratio_19= (mean_portfolio_19 - risk_free_return_19) / std_19
print(sharpe_ratio_19)

#balkendiagramm
y = [sharpe_ratio_08, sharpe_ratio_19]
x =[0.5, 1.5]
plt.figure(figsize=(10,7))
labels = "Portfolio 2008-2010", "Portfolio 2019-2021"
plt.ylim(-0.01,0.06)
plt.ylabel("Sharpe Ratio", fontsize = (15))
plt.title("Sharpe Ratio Portfolio 2008-2010 vs 2019-2021", fontsize = (25))
plt.bar(x, y, tick_label=labels, color=["darkgreen", "darkblue"])
plt.grid(which="major", color="k", linestyle="-.", linewidth=0.2)
plt.xticks(x, fontsize = (15))




