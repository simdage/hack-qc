import pandas as pd
from prophet import Prophet

from data_processing.data_cleaning import get_df, read_csv

# Python
# df_g = pd.read_csv('https://raw.githubusercontent.com/facebook/prophet/main/examples/example_yosemite_temps.csv')
# m = Prophet(changepoint_prior_scale=0.01).fit(df_g)
# future = m.make_future_dataframe(periods=300, freq='H')
# fcst = m.predict(future)
# fig = m.plot(fcst)
#
df_g1 = get_df()

precip = read_csv("dataset_csv/cas_1/precip.csv")
precip["Date"] = pd.to_datetime(precip["Date"])
df_g1 = pd.merge(df_g1, precip, on="Date", how="left")
df_g1["Date_H"] = df_g1['Date'].dt.round('H')


df_g = df_g1.groupby(['Date_H']).mean().reset_index()
df_g.loc[(df_g["Brutte_aval"] < 20), "Brutte_aval"] = None
df_g.loc[(df_g["Brutte_aval"] > 25), "Brutte_aval"] = None
df_g = df_g.rename(columns={"Date_H": "ds", "Brutte_aval": "y"})
df_g['Valeur'] = df_g['Valeur'].interpolate()
df_g['Valeur.1'] = df_g['Valeur.1'].interpolate()

model = Prophet(changepoint_prior_scale=0.01)
model.add_regressor("Brutte_quai")
model.add_regressor("Valeur")
model.add_regressor("Valeur.1")
model.fit(df_g)
print("model fitted")
forecast = model.predict(df = df_g[["ds", "Brutte_quai", "Valeur", "Valeur.1"]])
fig = model.plot(forecast)
