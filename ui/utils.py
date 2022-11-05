import pandas as pd
import copy

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

def get_errors(df, col_name):
    
    
    error_info = df.query(f'{col_name} == 1')

    response = pd.DataFrame(columns = ['Error', 'Start', 'Finish', 'Year', 'Month','Day'])
    old = 0
    start = True
    error = {'Error':col_name, 'Start':'', 'Finish':''}
    spike = copy.copy(error)
    for index in error_info.index:
        if start:
            if index != old+1: 
                old = index
                spike['Start']=df.loc[index]['Date']

                date = pd.to_datetime(df.loc[index]['Date'])
                spike["Year"] = date.year
                spike["Month"] = date.month
                spike["Day"] = date.day
                start = False
        else:
            if index != old+1:
                spike['Finish']= df.loc[old]['Date']
                # response = response.append(spike, ignore_index=True)

                spike = pd.DataFrame.from_dict([spike])
                response = pd.concat([response, spike], ignore_index=True)
                # pd.concat([response, pd.DataFrame(spike, columns=response.columns)], ignore_index=True)

                spike = copy.copy(error)
                spike['Start']=df.loc[index]['Date'] 
                date = pd.to_datetime(df.loc[index]['Date'])
                spike["Year"] = date.year
                spike["Month"] = date.month
                spike["Day"] = date.day

            old = index
    return response



if __name__ == "__main__":

    df = pd.read_csv('./corrected_df.csv')
    
    Random_Spikes = get_errors(df, 'Random_Spikes')
    Lower_outliers = get_errors(df, 'Lower_outliers')
    Upper_outliers = get_errors(df, 'Upper_outliers')

    full = pd.concat([Random_Spikes, Lower_outliers, Upper_outliers], ignore_index=True)
    full = full.sort_values(by=['Start'],ignore_index=True)
    full.to_csv('errors.csv')
