import os
import pandas as pd

def convert_all_files_to_csv(data_folder, 
                             delete_excl = False, 
                             case_list = ["cas1", "cas2", "cas3"]):
    for case in case_list:
        for excl_name in os.listdir(os.path.join(data_folder, case)):
            df = pd.read_excel(os.path.join(data_folder, case, excl_name))
            csv_name = excl_name.replace(".xlsx", ".csv")
            df.to_csv(os.path.join(data_folder, case, csv_name))
            if delete_excl:
                os.remove(path=os.path.join(data_folder, case, excl_name))
