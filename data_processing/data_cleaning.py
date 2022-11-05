import matplotlib

matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import polyfit


def read_csv(path):
    df = pd.read_csv(path, skiprows=2).iloc[:, 1:]
    return df


def get_df_cas_1():
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


def get_df_cas_2():
    df_ba_07_15_b = read_csv("dataset_csv/cas_2/gouin_so_2007_2015brutes.csv")
    df_ba_15_22_b = read_csv("dataset_csv/cas_2/gouin_so_2015_2022brutes.csv")

    df_ba_07_15_v = read_csv("dataset_csv/cas_2/gouin_so_2007_2015valide.csv")
    df_ba_15_22_v = read_csv("dataset_csv/cas_2/gouin_so_2015_2022valide.csv")

    df_ba_v = pd.concat([df_ba_07_15_v, df_ba_15_22_v])
    df_ba_b = pd.concat([df_ba_07_15_b, df_ba_15_22_b])

    df_ba_v = df_ba_v.rename(columns={"Valeur": "Validee_aval"})
    df_ba_b = df_ba_b.rename(columns={"Valeur": "Brutte_aval"})

    del df_ba_15_22_v, df_ba_15_22_b, df_ba_07_15_v, df_ba_07_15_b
    df_aval = pd.merge(df_ba_b, df_ba_v, on="Date")

    df_qu_07_15_b = read_csv("dataset_csv/cas_2/outardes_2_amont_2007_2015brutes.csv")
    df_qu_07_15_v = read_csv("dataset_csv/cas_2/outardes_2_amont_2007_2015-valide.csv")

    df_qu_15_22_b = read_csv("dataset_csv/cas_2/outardes_2_amont_2015_2022brutes.csv")
    df_qu_15_22_v = read_csv("dataset_csv/cas_2/outardes_2_amont_2015_2022valide.csv")

    df_qu_v = pd.concat([df_qu_07_15_v, df_qu_15_22_v])
    df_qu_b = pd.concat([df_qu_07_15_b, df_qu_15_22_b])

    del df_qu_15_22_v, df_qu_15_22_b, df_qu_07_15_v, df_qu_07_15_b

    df_qu_v = df_qu_v.rename(columns={"Valeur": "Validee_amont"})
    df_qu_b = df_qu_b.rename(columns={"Valeur": "Brutte_amont"})

    df_quai = pd.merge(df_qu_b, df_qu_v, on="Date")

    del df_ba_b, df_ba_v, df_qu_v, df_qu_b

    df = pd.merge(df_aval, df_quai, on="Date")

    del df_aval, df_quai

    df["Error_aval"] = df["Brutte_aval"] != df["Validee_aval"]
    df["Diff_aval"] = df["Brutte_aval"] - df["Validee_aval"]
    df["Error_amont"] = df["Brutte_amont"] != df["Validee_amont"]
    df["Diff_amont"] = df["Brutte_amont"] - df["Validee_amont"]

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["Hour"] = df["Date"].dt.hour
    print("df ready")
    return df


def get_df_cas_3():
    df_ba_07_10_b = read_csv("dataset_csv/cas_3/gouin_so_brute_2007_2010.csv")
    df_ba_10_15_b = read_csv("dataset_csv/cas_3/gouin_so_brute_2010_2015.csv")
    df_ba_15_22_b = read_csv("dataset_csv/cas_3/gouin_so_brute_2015_2022.csv")

    df_ba_07_10_v = read_csv("dataset_csv/cas_3/gouin_so_valide_2007_2010.csv")
    df_ba_10_15_v = read_csv("dataset_csv/cas_3/gouin_so_valide_2010_2015.csv")
    df_ba_15_22_v = read_csv("dataset_csv/cas_3/gouin_so_valide_2015_2022.csv")

    df_ba_v = pd.concat([df_ba_07_10_v, df_ba_10_15_v, df_ba_15_22_v])
    df_ba_b = pd.concat([df_ba_07_10_b, df_ba_10_15_b, df_ba_15_22_b])

    df_ba_v = df_ba_v.rename(columns={"Valeur": "Validee_aval"})
    df_ba_b = df_ba_b.rename(columns={"Valeur": "Brutte_aval"})

    del df_ba_07_10_v, df_ba_10_15_v, df_ba_15_22_v, df_ba_07_10_b, df_ba_10_15_b, df_ba_15_22_b

    df_aval = pd.merge(df_ba_b, df_ba_v, on="Date")

    df_qu_07_15_b = read_csv("dataset_csv/cas_3/outardes_2_amont_2007_2015brutes.csv")
    df_qu_07_15_v = read_csv("dataset_csv/cas_3/outardes_2_amont_2007_2015-valide.csv")

    df_qu_15_22_b = read_csv("dataset_csv/cas_3/outardes_2_amont_2015_2022brutes.csv")
    df_qu_15_22_v = read_csv("dataset_csv/cas_3/outardes_2_amont_2015_2022valide.csv")

    df_qu_v = pd.concat([df_qu_07_15_v, df_qu_15_22_v])
    df_qu_b = pd.concat([df_qu_07_15_b, df_qu_15_22_b])

    del df_qu_15_22_v, df_qu_15_22_b, df_qu_07_15_v, df_qu_07_15_b

    df_qu_v = df_qu_v.rename(columns={"Valeur": "Validee_amont"})
    df_qu_b = df_qu_b.rename(columns={"Valeur": "Brutte_amont"})

    df_quai = pd.merge(df_qu_b, df_qu_v, on="Date")

    del df_ba_b, df_ba_v, df_qu_v, df_qu_b

    df = pd.merge(df_aval, df_quai, on="Date")

    del df_aval, df_quai

    df["Error_aval"] = df["Brutte_aval"] != df["Validee_aval"]
    df["Diff_aval"] = df["Brutte_aval"] - df["Validee_aval"]
    df["Error_amont"] = df["Brutte_amont"] != df["Validee_amont"]
    df["Diff_amont"] = df["Brutte_amont"] - df["Validee_amont"]

    df["Date"] = pd.to_datetime(df["Date"])

    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day

    df["Hour"] = df["Date"].dt.hour
    print("df ready")
    return df


def polydiff(series, degree=4):
    X = [i % 365 for i in range(0, len(series))]
    y = series.values

    coef = polyfit(X, y, degree)
    print('Coefficients: %s' % coef)
    # create curve
    curve = list()
    for i in range(len(X)):
        value = coef[-1]
        for d in range(degree):
            value += X[i] ** (degree - d) * coef[d]
        curve.append(value)
    # create seasonally adjusted
    values = series.values
    diff = list()
    for i in range(len(values)):
        value = values[i] - curve[i]
        diff.append(value)
    return diff


def compute_exp_diff(diff, x):
    exp_diffs = list(np.zeros(x))
    for i, j in enumerate(diff[x:-x]):
        _diff = np.sum((j - diff[i - x:i + x]) ** 2)
        exp_diffs.append(_diff)
    exp_diffs += list(np.zeros(x))
    return np.asarray(exp_diffs)


def plot_graphs(exp_diff_, series, series_name="Brutte_aval"):
    fig, ax = plt.subplots(4)

    ax[0].set_title(f"Histogram of {series_name}")
    ax[0].hist(series, bins=50)

    ax[1].set_title(f"Trend of {series_name}")
    ax[1].plot(series)

    ax[2].set_title("Histogram of exponential differences")
    ax[2].hist(exp_diff_)

    ax[2].set_title("Trend of exponential differences")
    ax[3].plot(exp_diff_)


def replace_outliers_in_series(df, low_threshold=20, up_threshold=24, series="Brutte_aval"):
    low_threshold_ind = df[series][(df[series] < low_threshold)].index
    high_threshold_ind = df[series][(df[series] > up_threshold)].index

    indexes = {
        "low_threshold": low_threshold_ind,
        "high_threshold": high_threshold_ind

    }

    df.loc[low_threshold_ind, series] = np.nan
    df.loc[high_threshold_ind, series] = np.nan

    return df[series], indexes


def remove_spike_anomaly(series, pct_spike=0.03, plotgraphs_prior_filer=None, series_name=None):
    series = series.interpolate("linear")
    diff = polydiff(series, degree=4)
    exp_diff_1 = compute_exp_diff(diff, x=4)

    exp_diff_sorted = pd.Series(exp_diff_1).sort_values(ascending=True).values
    threshold = exp_diff_sorted[int(exp_diff_sorted.shape[0] * (1 - pct_spike))]
    above_threshold_ind = np.argwhere(exp_diff_1 > threshold).flatten().tolist()
    diff_zero_ind = series[exp_diff_1 == 0].index

    indexes = {
        "above_spike_threshold": above_threshold_ind,
        "diff_is_zero": diff_zero_ind

    }

    if plotgraphs_prior_filer:
        plot_graphs(exp_diff_1, series, series_name)

    series.loc[indexes["above_spike_threshold"]] = np.nan
    series.loc[indexes["diff_is_zero"]] = np.nan

    exp_diff_1[indexes["above_spike_threshold"]] = np.nan
    exp_diff_1[indexes["diff_is_zero"]] = np.nan

    return series, exp_diff_1, indexes


def clean_complete_series(df, low_threshold =20, high_threshold = 24, name = "Brutte_aval"):
    series, indexes_outliers = replace_outliers_in_series(df, low_threshold, high_threshold, name)
    series, exponential_diff_, indexes_spike = remove_spike_anomaly(series)
    series = interpolate_missing_data(series)
    indexes = {**indexes_outliers, **indexes_spike}

    return series, indexes


def interpolate_missing_data(series):
    series = series.interpolate(method='linear')
    return series


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



def clean_data_and_index_to_df(df, clean_series, index, name):
    len_ = len(df)
    df_corrected = pd.DataFrame(data={
        name: df[name].values,
        f"Corrected_{name}": clean_series,
        "Upper_outliers": np.zeros(len_),
        "Lower_outliers": np.zeros(len_),
        "Random_Spikes": np.zeros(len_),
        "Date": df["Date"]
    })

    df_corrected.loc[index["high_threshold"], "Upper_outliers"] = 1
    df_corrected.loc[index["low_threshold"], "Lower_outliers"] = 1
    df_corrected.loc[index['above_spike_threshold'], "Random_Spikes"] = 1

    df_corrected["Mistake_set_upper"] = identify_mistake_set(df_corrected, "Upper_outliers")
    df_corrected["Mistake_set_lower"] = identify_mistake_set(df_corrected, "Lower_outliers")
    df_corrected["Mistake_set_spike"] = identify_mistake_set(df_corrected, "Random_Spikes")
    return df_corrected




if __name__ == "__main__":
    df = get_df_cas_1()

    aval_clean, aval_index = clean_complete_series(df.copy(), 20, 24, "Brutte_aval")
    aval_corrected_beau = clean_data_and_index_to_df(df, aval_clean, aval_index, "Brutte_aval")
    aval_corrected_beau.to_csv("correct_df.csv", index = False)
    quai_clean, quai_index = clean_complete_series(df.copy(), 20, 24, "Brutte_quai")
    quai_corrected_beau = clean_data_and_index_to_df(df.copy(), quai_clean, quai_index, "Brutte_quai")

    df = get_df_cas_2()
    aval_clean, aval_index = clean_complete_series(df.copy(), 0, 3, "Brutte_aval")
    aval_corrected_outarde = clean_data_and_index_to_df(df.copy(), aval_clean, aval_index, "Brutte_aval")

    quai_clean, quai_index = clean_complete_series(df.copy(), 80, 88, "Brutte_amont")
    amont_corrected_outarde = clean_data_and_index_to_df(df.copy(), quai_clean, quai_index, "Brutte_amont")

    x = quai_clean[1300000: 1308000].values
    x1 = df['Brutte_quai'][1300000: 1308000].values
    plt.figure(figsize=(16, 8))
    plt.title("Série corrigée vs. série brutte")
    plt.plot(x, alpha=1, label="Série corrigée")
    plt.plot(x1, alpha=0.3, color="red", label="Série brutte")
    plt.legend()
    plt.show()
