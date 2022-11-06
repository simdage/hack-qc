import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM

from data_processing.data_cleaning import get_df_cas_1, clean_complete_series, clean_data_and_index_to_df

df = get_df_cas_1()
aval_clean, aval_index = clean_complete_series(df.copy(), 20, 24, "Brutte_aval")
aval_corrected_beau = clean_data_and_index_to_df(df, aval_clean, aval_index, "Brutte_aval")


quai_clean, quai_index = clean_complete_series(df.copy(), 20, 24, "Brutte_quai")
quai_corrected_beau = clean_data_and_index_to_df(df.copy(), quai_clean, quai_index, "Brutte_quai")



df_series = pd.concat([aval_clean, quai_clean], axis=1)
df_series["Month"] = df["Date"].dt.month

df1 = df_series.dropna()
df1['Diff'] = df1["Brutte_aval"] - df1["Brutte_quai"]
df1["Diff_exp"] = df1["Diff"] ** 2

df1["lag_Diff"] = df1["Diff"].shift(1)
df1["lag_Diff_exp"] = df1["Diff_exp"].shift(1)
df1 = df1.dropna()
anomaly_inputs = ['Diff_exp', 'Diff', "lag_Diff", "lag_Diff_exp", "Brutte_aval", "Brutte_quai", "Month"]

model_IF = IsolationForest(contamination=0.15, random_state=42, n_estimators=800)
model_IF.fit(df1[anomaly_inputs])
print("model fitted")
df1['anomaly_scores_IF'] = model_IF.decision_function(df1[anomaly_inputs])
df1['anomaly_scores_IF'].argmax()


plt.figure()
plt.hist(df1["anomaly_scores_IF"], bins = 50)
plt.show()



import shap

exp = shap.TreeExplainer(model_IF) #Explainer
sample = df1[anomaly_inputs].sample(4000)
shap_values = exp.shap_values(sample)  #Calculate SHAP values
# shap.initjs()

plt.figure()
shap.summary_plot(shap_values, sample, plot_type="bar")

plt.figure()
shap.summary_plot(shap_values, sample)
plt.set_yticklabels(anomaly_inputs)


import matplotlib.pyplot as plt
f = plt.figure()
shap.summary_plot(shap_values, sample)
f.savefig("/summary_plot1.png", bbox_inches='tight', dpi=600)



plt.figure()
shap.bar_plot(exp.expected_value,df1[anomaly_inputs].iloc[134000,:] ,feature_names = anomaly_inputs )

plt.figure(figsize=(12,8))
plt.title("Valeurs explicatives pour une prédiction d'anomalie")
shap.bar_plot(shap_values[1041],features =sample.iloc[1041,:] ,feature_names =sample.columns )
plt.yticks(list(range(1, 8)), ["Month = 1", "Lag_Diff_exp = 0.22", "Diff_exp = 0.17", "lag_Diff = 0.47", "Brute_aval = 22.12",
                            "Brute_quai = 21.7", "Diff = 0.42"] )

plt.show()





fig, ax = plt.subplots(3)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores_IF"][1300000: 1305000] * -1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][1300000: 1305000], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][1300000: 1305000], label="Beauharnois quai")

ax[2].set_title("Distribution des scores_IF d'anomalie")
ax[2].hist(df1["anomaly_scores_IF"] * -1, bins=50)

fig, ax = plt.subplots(2)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores_IF"][567776: 573776] * -1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][567776: 573776], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][567776: 573776], label="Beauharnois quai")

model_svm = OneClassSVM()
model_svm.fit(df1[anomaly_inputs])
df1['anomaly_scores_SVM'] = model_svm.decision_function(df1[anomaly_inputs])

fig, ax = plt.subplots(2)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores_IF"][1300000: 1305000] * -1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][1300000: 1305000], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][1300000: 1305000], label="Beauharnois quai")


fig, ax = plt.subplots(2)


ax[1].plot(df1["anomaly_scores_IF"][131000: 137000] * -1, label = "Score d'anomalie")
ax[1].legend()


ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][131000: 137000], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][131000: 137000], label="Beauharnois quai")
ax[0].legend()




df = pd.merge(quai_corrected_beau, df1[["anomaly_scores_IF"]], left_index=True, right_index=True)
# df_a.to_csv("df_corrected.csv", index = False)
df_a = pd.read_csv("df_corrected.csv")


df_a["Anomaly_IF"] = df_a["anomaly_scores_IF"] < -0.05
df_a["Anomaly_IF"].sum()

df_a["Anomaly"] = df_a[["Upper_outliers", "Lower_outliers", "Random_Spikes", "Anomaly_IF"]].sum(axis =1)
df_a["Anomaly"] = df_a["Anomaly"] >= 1

df = pd.merge(df_a, df[["Brutte_quai"]], left_index=True, right_index=True)

df["Brute_quai_nan"] = df["Brutte_quai"].copy()
import numpy as np
df.loc[df["Anomaly"] == 1, "Brute_quai_nan"] = np.nan

from prophet import Prophet
df["Date"] = pd.to_datetime(df["Date"])
df["Date_H"] = df['Date'].dt.round('H')
df_g = df.groupby(['Date_H']).mean().reset_index()
df_g = df_g.rename(columns={"Date_H": "ds", "Brute_quai_nan": "y"})
model = Prophet(changepoint_prior_scale=0.01)
model.add_regressor("Corrected_Brutte_aval")
model.fit(df_g)
print("model fitted")
forecast = model.predict(df = df_g[["ds", "Corrected_Brutte_aval"]])
model.plot(forecast)
df = pd.merge(df, forecast, left_on="Date_H", right_on="ds")

np.mean(np.abs(df["yhat"] - df['Brutte_quai']))
np.mean(np.abs(df[df["Anomaly"]!=1]["yhat"] - df[df["Anomaly"]!=1]['Corrected_Brutte_quai']))





import numpy as np
from sklearn.linear_model import LinearRegression

df_reg = df.dropna()

reg = LinearRegression().fit(df_reg[(df_reg["Anomaly"]!=1) ][["Corrected_Brutte_aval"]], df_reg[df_reg["Anomaly"]!=1]["Corrected_Brutte_quai"])


reg.score(df_reg[(df_reg["Anomaly"]!=1) ][["Corrected_Brutte_aval"]], df_reg[df_reg["Anomaly"]!=1]["Corrected_Brutte_quai"])
df["pred"] = reg.predict(df[["Corrected_Brutte_aval"]])
np.mean(np.abs(df["pred"] - df['Corrected_Brutte_aval']))
np.max(np.abs(df["pred"] - df['Corrected_Brutte_aval']))
np.min(np.abs(df["pred"] - df['Corrected_Brutte_aval']))





df[["Brutte_aval", "Brutte_quai"]].corr()
df[df["Anomaly"] != 1][["Brutte_aval", "Corrected_Brutte_quai"]].corr()

plt.figure()
plt.scatter(df["Brutte_aval"], df["Brutte_quai"])
plt.show()

plt.figure()
plt.scatter(df[df["Anomaly"] != 1]["Brutte_aval"], df[df["Anomaly"] != 1]["Corrected_Brutte_quai"])
plt.show()



df[df["Upper_outliers"] == 1]

df[df["Anomaly"] != 1]