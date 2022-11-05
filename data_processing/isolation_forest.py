import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
import matplotlib.pyplot as plt
from data_processing.data_cleaning import get_df_cas_1, clean_complete_series, read_csv
import numpy as np

df = get_df_cas_1()
aval_clean, aval_index = clean_complete_series(df.copy(), 20, 24, "Brutte_aval")
quai_clean, quai_index = clean_complete_series(df.copy(), 20, 24, "Brutte_quai")

precip = read_csv("dataset_csv/cas_1/precip.csv")

df_series = pd.concat([aval_clean, quai_clean], axis=1)
df_series["Date"] = df["Date"].copy()

precip["Date"] = pd.to_datetime(precip["Date"])
df_series = pd.merge(df_series, precip, on="Date", how="left")
df_series.loc[:, ["Valeur", "Valeur.1"]] = df_series[["Valeur", "Valeur.1"]].interpolate()

for j in ["Brutte_aval", "Brutte_quai", "Valeur", "Valeur.1"]:
    for i in [1]:
        df_series.loc[:, f"{j}_lag_{i}"] = df_series[j].shift(i)
        df_series.loc[:, f"{j}_post_{i}"] = df_series[j].shift(-i)

# df_series['Diff_lag_1'] = (df_series["Brutte_aval_lag_1"] - df_series["Brutte_quai_lag_1"])**2
# df_series['Diff_post_1'] = (df_series["Brutte_aval_post_1"] - df_series["Brutte_quai_post_1"])**2

df_series["Difference_Exp"] = (df["Brutte_aval"] - df["Brutte_quai"])**2
df_series["Difference"] = (df["Brutte_aval"] - df["Brutte_quai"])

df1 = df_series.dropna()

anomaly_inputs = [i for i in df1.columns if i.startswith(("Diff"))]
model_IF = IsolationForest(random_state=42, n_estimators=300, contamination=0.05)
model_IF.fit(df1[anomaly_inputs])
df1['anomaly_scores_IF'] = model_IF.decision_function(df1[anomaly_inputs])

model_svm = OneClassSVM()
model_svm.fit(df1[anomaly_inputs])
df1['anomaly_scores_SVM'] = model_svm.decision_function(df1[anomaly_inputs])



# df1['anomaly'] = model_IF.predict(df1[anomaly_inputs])


df1["anomaly_scores"].argmax()

fig, ax = plt.subplots(3)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores_IF"][1300000: 1305000] * -1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][1300000: 1305000], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][1300000: 1305000], label="Beauharnois quai")
ax[0].scatter(df1[1300000: 1305000][df1["anomaly_scores_IF"] < -0.1625].index,
              df1[1300000: 1305000][df1["anomaly_scores_IF"] < -0.1625]["Brutte_aval"], color="red", label="anomalie")
ax[0].legend()

ax[2].set_title("Distribution des scores_IF d'anomalie")
ax[2].hist(df1["anomaly_scores_IF"] * -1, bins=50)

df1["anomaly_scores_IF"].argmax()

df1[df1["anomaly_scores_IF"] < -0.155] = np.nan

fig, ax = plt.subplots(3)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores_IF"][120029 - 2000: 120029 + 20000] * -1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][120029 - 2000: 120029 + 20000], label="Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][120029 - 2000: 120029 + 20000], label="Beauharnois quai")
ax[0].scatter(df1[120029 - 2000: 120029 + 20000][df1["anomaly_scores_IF"] < -0.1625].index,
              df1[120029 - 2000: 120029 + 20000][df1["anomaly_scores_IF"] < -0.1625]["Brutte_aval"], color="red",
              label="anomalie")
ax[0].legend()

ax[2].set_title("Distribution des scores_IF d'anomalie")
ax[2].hist(df1["anomaly_scores_IF"] * -1, bins=50)

df1[df1["anomaly_scores_IF"] < -0.155] = np.nan
