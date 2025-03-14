import csv
import os

import pandas as pd

# csv_name is the name of the file you want to read
def csv_read(csv_name):
    result = []
    with open(csv_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            result.append(row)
    return result
            

# file_name is the name you want for the file
# arr is an 2d array of rows, index 0 is headers
def csv_write(file_name, arr):
    with open(file_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(arr)

# for getting the current path, if needed
def get_path():
    print(os.getcwd())

columns = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 'd',
       'sales', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
       'event_name_1', 'event_type_1', #'event_name_2', 'event_type_2',
       'snap_CA', 'snap_TX', 'snap_WI', 'sell_price']
categoricals = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]#, "event_name_2", "event_type_2"
encodeables = ["dept_id", "cat_id", "state_id", "event_name_1", "event_type_1"]
ints = ['sales', 'wm_yr_wk', 'wday', 'month', 'year', "snapCA", "snapTX", "snapWI", "sell_price"]
other = ['d', 'date', 'weekday']
drop = ["d", "date", "event_type_2", "event_name_2", "wm_yr_wk"]


def fill_nulls(df):
    flags = {}
    for col in columns:
        # add categorical variable for null
        if (df[col].isnull().any()) :
            df[col+"_filled"] = df[col].isnull().astype(int)
        else:
            continue

        if col in categoricals:
            df[[col]] = df[[col]].fillna(value=("no_"+col))
        elif col in ints:
            df[[col]] = df[[col]].fillna(value=df[[col]].mean())#adjust to calculate mean
    return df

def encode(df):
    for en in encodeables:
        print("cat:",en)
        onehot = pd.get_dummies(df[en]).add_prefix(en+":")
        print(onehot)
        df = df.join(onehot)
    df.drop(columns=encodeables,inplace=True)
    return df




# example method usage
'''get_path()
test = [["name", "id"],["adam","1"],["eve","2"]]
csv_write("test.csv",test)
print(csv_read("test.csv"))'''

df = pd.read_csv("merged_df.csv")
df.drop(columns=["event_name_2","event_type_2"], inplace=True)
print(df.head())
df = fill_nulls(df).head()
print(df)
print(df.dtypes)
print(encode(df).dtypes)
#print(df['sell_price'].isnull().sum())
#print(df.groupby(["d","sell_price"]).count())