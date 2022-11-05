import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import itertools
import warnings
from prophet import Prophet
import matplotlib.pyplot as plt
from numpy import polyfit


def read_csv(path):
    df = pd.read_csv(path, skiprows=2).iloc[:, 1:]
    return df


def get_df():
    df_ba_07_15_b = read_csv("dataset_csv/cas_1/beauharnois_aval_2007_2015_brutes.csv")
    df_ba_15_22_b = read_csv("dataset_csv/cas_1/beauharnois_aval_2015_2022_brutes.csv")

    df_ba_07_15_v = read_csv("dataset_csv/cas_1/beauharnois_aval_2007_2015_validees.csv")
    df_ba_15_22_v = read_csv("dataset_csv/cas_1/beauharnois_aval_2015_2022_validees.csv")

    df_ba_v = pd.concat([df_ba_07_15_v, df_ba_15_22_v])
    df_ba_b = pd.concat([df_ba_07_15_b, df_ba_15_22_b])

    df_ba_v = df_ba_v.rename(columns={"Valeur": "Validee_aval"})
    df_ba_b = df_ba_b.rename(columns={"Valeur": "Brutte_aval"})

    del df_ba_15_22_v, df_ba_15_22_b, df_ba_07_15_v, df_ba_07_15_b
    df_aval = pd.merge(df_ba_b, df_ba_v, on="Date")

    df_qu_07_15_b = read_csv("dataset_csv/cas_1/quai_de_beauharnois_2007_2015_brutes.csv")
    df_qu_07_15_v = read_csv("dataset_csv/cas_1/quai_de_beauharnois_2007_2015_validees.csv")

    df_qu_15_22_b = read_csv("dataset_csv/cas_1/quai_de_beauharnois_2015_2022_brutes.csv")
    df_qu_15_22_v = read_csv("dataset_csv/cas_1/quai_de_beauharnois_2015_2022_validees.csv")

    df_qu_v = pd.concat([df_qu_07_15_v, df_qu_15_22_v])
    df_qu_b = pd.concat([df_qu_07_15_b, df_qu_15_22_b])

    del df_qu_15_22_v, df_qu_15_22_b, df_qu_07_15_v, df_qu_07_15_b

    df_qu_v = df_qu_v.rename(columns={"Valeur": "Validee_quai"})
    df_qu_b = df_qu_b.rename(columns={"Valeur": "Brutte_quai"})

    df_quai = pd.merge(df_qu_b, df_qu_v, on="Date")

    del df_ba_b, df_ba_v, df_qu_v, df_qu_b

    df = pd.merge(df_aval, df_quai, on="Date")

    del df_aval, df_quai

    df["Error_aval"] = df["Brutte_aval"] != df["Validee_aval"]
    df["Diff_aval"] = df["Brutte_aval"] - df["Validee_aval"]
    df["Error_quai"] = df["Brutte_quai"] != df["Validee_quai"]
    df["Diff_quai"] = df["Brutte_quai"] - df["Validee_quai"]

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["Hour"] = df["Date"].dt.hour
    print("df ready")
    return df


df = get_df()

plt.figure(figsize=(16, 10))
plt.title("Distribution des Bruttes bruttes et validees")
sns.histplot(data=df[(df["Brutte_aval"] > 15) & (df["Brutte_aval"] < 45)], x="Brutte_aval", hue="Error_aval", bins=100,
             stat="density", common_norm=False)
plt.show()

plt.figure(figsize=(16, 10))
plt.title("Distribution des Bruttes bruttes et validees")
sns.histplot(data=df[(df["Brutte_quai"] > 18) & (df["Brutte_quai"] < 24)], x="Brutte_quai", hue="Error_quai", bins=100,
             stat="density", common_norm=False)
plt.show()

plt.figure(figsize=(16, 10))
sns.violinplot(data=df[(df["Diff_aval"] > -3) & (df["Diff_aval"] < 3)], x="Year", y="Diff_aval")
plt.show()

plt.figure(figsize=(16, 10))
sns.violinplot(data=df[(df["Diff_aval"] > -3) & (df["Diff_aval"] < 3)], x="Month", y="Diff_aval")
plt.show()

plt.figure(figsize=(16, 10))
sns.violinplot(data=df[(df["Diff_aval"] > -1) & (df["Diff_aval"] < 1)], x="Hour", y="Diff_aval")
plt.show()

precip = pd.read_csv("dataset_csv/cas_1/beauharnois_aval_2007_2015_brutes.csv")

plt.figure()
plt.hist(df[(df["Error_aval"] == 1) & (df["Diff_aval"] < 50) & (df["Diff_aval"] > -10)]["Diff_aval"], bins=50)
plt.show()

plt.figure()
plt.hist(df[(df["Error_quai"] == 1) & (df["Diff_quai"] < 4) & (df["Diff_quai"] > -4)]["Diff_quai"], bins=50)
plt.show()


# fig = px.line(df[df["Error"] == 1].iloc[:1000], "Date", "Brutte")
# fig1 = px.line(df[df["Error"] == 1].iloc[:1000], "Date", "Validee")
# fig.add_trace(fig1.data[0])
# fig.show()


def identify_mistake_set(df, error_col):
    mistake_sets = [0]
    set = 0
    for i, j in enumerate(list(df[error_col][1:])):
        if j == 1 and df[error_col][i - 1] == 0:
            set += 1
            mistake_sets.append(set)
        if j == 1 and df[error_col][i - 1] == 1:
            mistake_sets.append(set)
        if j == 0:
            mistake_sets.append(0)

    return mistake_sets


import plotly.express as px

mistake_set = identify_mistake_set(df, "Error_aval")
df["Mistake_set_aval"] = mistake_set

mistake_set = identify_mistake_set(df, "Error_quai")
df["Mistake_set_quai"] = mistake_set

for i in [3500, 4500]:
    indexes = df[df["Mistake_set_aval"] == i].index
    plt.figure()
    plt.title("Comparaison des séries validées et brutes pour Beauharnois")
    plt.plot(df.iloc[indexes.min() - 2000:indexes.max() + 2000, ]["Brutte_aval"], color = "red", label = "Brute_aval")
    plt.plot(df.iloc[indexes.min() - 2000:indexes.max() + 2000, ]["Validee_aval"], color = "green", label = "Validee_aval")
    plt.plot(df.iloc[indexes.min() - 2000:indexes.max() + 2000,]["Brutte_quai"], color = "red", label ="Brute_quai")
    plt.plot(df.iloc[indexes.min() - 2000:indexes.max() + 2000, ]["Validee_quai"], color = 'blue', label = "Validee_quai")
    plt.legend()
    plt.show()


    fig = px.line(df.iloc[indexes.min() - 2000:indexes.max() + 2000, :], "Date",
                  y=["Brutte_aval", "Validee_aval", "Brutte_quai", "Validee_quai"],
                  color_discrete_map={
                      "Brutte_quai": "red",
                      "Validee_quai": "green",
                      "Brutte_aval": "red",
                      "Validee_aval": "blue",
                  })
    fig.show()

from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Common code for display result
def show_graph(df1, df2, title):
    data = pd.concat([df1, df2])
    data.reset_index(inplace=True, drop=True)
    for col in data.columns:
        if col.lower().startswith('pred'):
            data[col].plot(label=col, linestyle="dotted")
        else:
            data[col].plot(label=col)
    plt.title(title)
    plt.legend()
    plt.show()


def VARMAX_model(df_train, df_test):
    # fit model
    model = SARIMAX(endog=df_train["Brutte_aval"], exog=df_train, order=(1, 0, 0), trend='n')
    model_fit = model.fit()
    # make prediction
    yhat = model_fit.get_forecast(steps=499, exog=df_test.iloc[:-1, :]).summary_frame(alpha=0.05)
    res = pd.DataFrame({"Pred": yhat.values, "Y": df_test["Brutte_aval"][1:]})
    return res


# fit polynomial: x^2*b1 + x*b2 + ... + bn

plt.figure()
plt.plot(series)
plt.show()

diff = polydiff(series)

plt.figure()
plt.plot(diff)
plt.show()

exp_diff_ = exp_diff(diff, 4)
# plot_graphs(exp_diff_, series)
exp_diff_sorted = np.sort(exp_diff_)

treshold = exp_diff_sorted[int(exp_diff_sorted.shape[0] * 0.98)]
series.loc[(exp_diff_ > treshold)] = np.nan
series.loc[exp_diff_ == 0] = np.nan
exp_diff_[(exp_diff_ > treshold)] = np.nan
exp_diff_[exp_diff_ == 0] = np.nan

series.isna().sum()

exp_diff_[np.isnan(exp_diff_) == False].min()

plot_graphs(exp_diff_, series)

plt.figure()
plt.plot(series)
plt.show()

x = series[1300000: 1308000].values
plt.figure()
plt.plot(x)
plt.show()

x = series[1500000: 1508000].values
x1 = df['Brutte_aval'][1500000: 1508000].values
plt.figure()
plt.title("Détection anomalie locale: Série corrigée vs. série brute")
plt.plot(x, alpha=1, label="Série corrigée")
plt.plot(x1, alpha=0.3, color="red", label="Série brute")
plt.legend()
plt.savefig("anomalie_locale.png")
plt.show()

x = series[1300000: 1308000].values
x1 = df['Brutte_aval'][1300000: 1308000].values
plt.figure(figsize=(16, 8))
plt.title("Série corrigée vs. série brute")
plt.plot(x, alpha=1, label="Série corrigée")
plt.plot(x1, alpha=0.3, color="red", label="Série brutte")
plt.legend()
plt.show()


x = series[2000: 3000].values
x1 = df['Brutte_aval'][2000: 3000].values
plt.figure(figsize=(16, 8))
plt.title("Série corrigée vs. série brute")
plt.plot(x, alpha=1, label="Série corrigée")
plt.plot(x1, alpha=0.3, color="red", label="Série brutte")
plt.savefig("anomalie_locale1.png")
plt.legend()
plt.show()

plt.figure()
plt.plot(x)
plt.show()

np.argmin(x)

x.min()

y = exp_diff_[1307000: 1308000]
plt.figure()
plt.plot(x)
plt.show()
1307000 - 1336526
plt.figure()
plt.plot(exp_diff_[1307000: 1308000])
plt.show()
series = series.dropna()
plot_graphs(exp_diff_, series)

max(exp_diff_)

diff = polydiff(series)
exp_diff_ = exp_diff(diff)
# plot_graphs(exp_diff_, series)
series.loc[np.asarray(exp_diff_) > 0.8] = None
series = series.dropna()
# plot_graphs(exp_diff_, series)


exp_diff_sorted = np.sort(exp_diff_)
exp_diff_sorted[int(exp_diff_sorted.shape[0] * 0.99)]

diff = polydiff(series)
exp_diff_ = exp_diff(diff)
plot_graphs(exp_diff_, series)
series.loc[np.asarray(exp_diff_) > 0.5] = None
series = series.dropna()
plot_graphs(exp_diff_, series)

plt.plot(series)
plt.show()

series = series[np.asarray(exp_diff_) < 3.21]

# compute mse between value and two values surrounding


df[(df["Brutte_aval"] > 30) & (df["Brutte_aval"] < 20)]["Brutte_aval"] = None
df['Brutte_aval_smoothed'] = df['Brutte_aval'].rolling(30).mean()

# removing all the NULL values using
# dropna() method
df.dropna(inplace=True)

df = df.groupby(['Year', 'Month', 'Day', "Hour"]).mean()
df = df.reset_index()
df["Date"] = df["Year"].astype(int).astype(str) + "-" + df["Month"].astype(int).astype(str) + "-" + df["Day"].astype(
    int).astype(str)
df["Date"] = pd.to_datetime(df["Date"])
df.loc[(df["Brutte_aval"] < 20), "Brutte_aval"] = None
df.loc[(df["Brutte_aval"] > 25), "Brutte_aval"] = None

y = df[["Date", "Brutte_quai", "Brutte_aval"]]
y = y.rename(columns={"Date": "ds", "Brutte_aval": "y"})

model = Prophet(changepoint_prior_scale=0.01)
model.fit(y)
print("model fitted")
forecast = model.make_future_dataframe(periods=1, freq='H')
forecast = model.predict(forecast)
fig = model.plot(forecast)

brutte_aval_forecast = model.predict(forecast)
brutte_aval_forecast = pd.merge(brutte_aval_forecast, df[["Date", "Brutte_aval"]], left_on="ds", right_on="Date")

plt.figure()
plt.plot(brutte_aval_forecast[5600:]["Brutte_aval"], label="Brutte")
plt.plot(brutte_aval_forecast[5600:]["yhat"], label="Pred")
plt.plot(brutte_aval_forecast[5600:]["yhat_lower"], label="Pred_low")
plt.plot(brutte_aval_forecast[5600:]["yhat_upper"], label="Pred_upper")
plt.show()

plt.figure()
plt.plot(brutte_aval_forecast["Brutte_aval"], label="Brutte")
plt.plot(brutte_aval_forecast["yhat"], label="Pred")
plt.plot(brutte_aval_forecast["yhat_lower"], label="Pred_low")
plt.plot(brutte_aval_forecast["yhat_upper"], label="Pred_upper")
plt.show()

df = get_df()
df["Date_H"] = df['Date'].dt.round('D')
df = pd.merge(df, forecast[["yhat", "ds"]], left_on="Date_H", right_on="ds")

diff = df["Brutte_aval"] - df["yhat"]

plt.figure()
plt.plot(diff)
plt.show()

df["Date"] = pd.to_datetime(df["Date"])

df_pred = pd.merge(df, forecast[5600:5601], left_on="Date", right_on="ds")

plt.figure()
plt.plot(df_pred["Brutte_aval"], label="Brutte")
plt.plot(df_pred["yhat"], label="Pred")
plt.plot(df_pred["yhat_lower"], label="Pred_low")
plt.plot(df_pred["yhat_upper"], label="Pred_upper")
plt.show()

import numpy as np
from tqdm import tqdm

smoothed_data = []
for date in tqdm(df['Date']):
    df['gkv'] = np.exp(
        -(((df['Date'] - date).apply(lambda x: x.days)) ** 2) / (2 * (2 ** 2))
    )
    df['gkv'] /= df['gkv'].sum()
    smoothed_data.append(round(df['Brutte_aval'] * df['gkv']).sum())

df['Brutte_aval_smoothed'] = smoothed_data

import numpy as np

fwhm = 5  # in ms
k = 50
gauss_time = 2000 * np.arange(-k, k) / 2000
# create Gaussian window
gauswin = np.exp(-(4 * np.log(2) * gauss_time ** 2) / fwhm ** 2)
plt.plot(gauswin)

# initialize filtered signal vector

n = len(df)

filtSig_Gauss = np.zeros(n)
signal = df["Brutte_aval"].values

# # implement the running mean filter
for i in range(k + 1, n - k - 1):
    # each point is the weighted average of k surrounding points
    filtSig_Gauss[i] = np.sum(signal[i - k:i + k] * gauswin)

plt.plot(signal, 'r', label='Original')
plt.plot(filtSig_Gauss, 'k', label='Gaussian-filtered')
plt.xlabel('Time (s)')
plt.ylabel('amp. (a.u.)')
plt.legend()
plt.title('Gaussian smoothing filter')

# todo use mean by hour
