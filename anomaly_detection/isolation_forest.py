
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
from data_processing.data_cleaning import get_df_cas_1, clean_complete_series
import numpy as np
df = get_df_cas_1()
aval_clean, aval_index = clean_complete_series(df.copy(), 20, 24, "Brutte_aval")
quai_clean, quai_index = clean_complete_series(df.copy(), 20, 24, "Brutte_quai")



df_series = pd.concat([aval_clean, quai_clean], axis = 1)

df1 = df_series.dropna()

anomaly_inputs = ['Brutte_aval', 'Brutte_quai']
model_IF = IsolationForest(contamination=0.1, random_state=42)
model_IF.fit(df1[anomaly_inputs])
df1['anomaly_scores'] = model_IF.decision_function(df1[anomaly_inputs])
df1['anomaly'] = model_IF.predict(df1[anomaly_inputs])
df1[df1["anomaly_scores"] < -0.155] = np.nan

df_series = pd.merge(df, df)



plt.figure()
plt.hist(df1["anomaly_scores"])
plt.show()



fig, ax = plt.subplots(3)

ax[1].set_title(f"Score anomalie détectée")
ax[1].plot(df1["anomaly_scores"][1300000: 1305000]*-1)

ax[0].set_title("Série temporelle Beauharnois")
ax[0].plot(df1["Brutte_aval"][1300000: 1305000], label = "Beauharnois aval")
ax[0].plot(df1["Brutte_quai"][1300000: 1305000], label = "Beauharnois quai")


ax[2].set_title("Distribution des scores d'anomalie")
ax[2].hist(df1["anomaly_scores"]*-1, bins = 50)
