import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
import statsmodels.api as sm
import itertools
import warnings


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


if __name__ == "__main__":
    # file_name = 'quai_de_beauharnois_2007_2015_brutes.csv'
    # path = '../dataset_csv/cas_1/'

    # df = pd.read_csv(os.path.join(path, file_name),header=2)
    # info,data = process_df(df)
    # print(data)

    data = get_df()
    daily = data.query('Year == 2007 and Month == 1 and Day == 1')
    plt.plot(daily['Brutte_aval'])
    plt.ylim([20,25])
    plt.show()