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
# drop weekday, dept_id?
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
    # encode categorical variables
    for en in encodeables:
        #print("cat:",en)
        onehot = pd.get_dummies(df[en]).add_prefix(en+":").astype(int)
        #print(onehot)
        df = df.join(onehot)
    df.drop(columns=encodeables,inplace=True)

    # encode day to a number
    df["d"] = df["d"].apply(lambda x: int(x[2:]))
    return df


def add_cols(df):
    # adding last couple days data
    right = df[["item_id", "store_id", "d", "sales"]].rename(columns={"d":"last"})
    last = "d"

    remove = []
    for day in range(28):
        df["last"] = df[last].apply(lambda x: x - 1)
        last = "last"

        df = df.merge(right, how="left", on=["item_id","store_id","last"], suffixes=("","_"+str(day)))
        remove.append("sales_"+str(day))
    
    df["item_store_last_day_sales"] = df["sales_0"]
    df.to_csv("test.csv")
    df["item_store_last_day_sales_filled"] = df["sales_0"].isnull().astype(int) # mark null days
    df["item_store_last_day_sales"] = df["item_store_last_day_sales"].fillna(value=df["item_store_last_day_sales"].mean(skipna=True)) # fill null days with mean
    
    for med in [7, 14, 21, 28]:
        cols = ["sales_"+str(i) for i in range(med)]
        #print(df[cols].median())
        df["item_store_L"+str(med)+"d_day_median_sales"] = df[cols].median(skipna=False)
        df["item_store_L"+str(med)+"d_day_median_sales_filled"] = df[cols].median(skipna=False).isnull().astype(int)
        #print(df.head())

    df.drop(columns = remove,inplace=True)
    #print(df.isnull().any())
    #print(df.head())

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
df = encode(df)
print(df.dtypes)
df = add_cols(df)
print(df.dtypes)
print(df.isnull().any())
print(df.notnull().any())
#print(df['sell_price'].isnull().sum())
#print(df.groupby(["d","sell_price"]).count())