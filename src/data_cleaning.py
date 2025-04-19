import csv
import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
       'sales', 'event_name_1', 'event_type_1', 'snap_CA', 'sell_price'] # all columns that are not in early_drop

categoricals = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id", "event_name_1", "event_type_1"]

one_hot_encodeables = ["dept_id", "store_id", "cat_id", "state_id", "event_name_1", "event_type_1"]
enum = ["item_id"]

ints = ['sales', 'wm_yr_wk', 'wday', 'month', 'year', "snapCA", "sell_price"]
early_drop = ["weekday", "event_type_2", "event_name_2", "wm_yr_wk", 'snap_TX', 'snap_WI'] #everything that we drop right away
late_drop = ["date"] # everything that is eventually dropped, but is necessary for calculation
# keep "d" but don't use as feature


def fill_nulls(df):
    flags = {}
    remove = []
    for col in columns:
        # add categorical variable for null
        if (df[col].isnull().any()) :
            df[col+"_filled"] = df[col].isnull().astype(int)
            remove.append(col)
        else:
            continue

        if col in categoricals:
            df[[col]] = df[[col]].fillna(value=("no_"+col))
        elif col in ints:
            df[[col]] = df[[col]].fillna(value=df[[col]].mean())#adjust to calculate mean
    df.drop(remove, axis=1)
    return df

def encode(df):
    # encode categorical variables
    for en in one_hot_encodeables:
        #print("cat:",en)
        onehot = pd.get_dummies(df[en]).add_prefix(en+":").astype(int)
        #print(onehot)
        df = df.join(onehot)
    #df.drop(columns=one_hot_encodeables,inplace=True)

    # encode day to a number
    df["d"] = df["d"].apply(lambda x: int(x[2:]))

    # enumerate these values for enmbedding
    for col in enum:
        vocab = sorted(list(set(df[col])))
        word_to_index = {word: i for i, word in enumerate(vocab)}
        
        df[col] = [word_to_index[x] for x in df[col]]
    #df.drop(columns=enum, inplace=True)

    return df


def add_cols(df):
    # adding last couple days data
    right = df[["item_id", "store_id", "d", "sales"]].rename(columns={"d":"last"})
    last = "d"

    remove = ["last"]
    for day in range(29):
        df["last"] = df[last].apply(lambda x: x - 1)
        last = "last"

        df = df.merge(right, how="left", on=["item_id","store_id","last"], suffixes=("","_"+str(day)))
        remove.append("sales_"+str(day))
    print(df.shape)
    
    df["item_store_last_day_sales"] = df["sales_0"]
    df["item_store_last_day_sales_filled"] = df["sales_0"].isnull().astype(int) # mark null days
    df["item_store_last_day_sales"] = df["item_store_last_day_sales"].fillna(value=df["item_store_last_day_sales"].mean(skipna=True)) # fill null days with mean
    
    for med in [7, 14, 21, 28]:
        cols = ["sales_"+str(i) for i in range(med - 6, med + 1)]
        df["item_store_L"+str(med)+"d_day_median_sales_filled"] = df[cols].isnull().any(axis=1).astype(int)
        df["item_store_L"+str(med)+"d_day_median_sales"] = df[cols].median(axis="columns",skipna=False)
        #print(df.head())
    
    # trim first 30 rows
    #print(df.shape)
    df = df.dropna(subset=["date"])
    #print(df.shape)

    
    df.drop(columns = remove,inplace=True)
    #print(df.isnull().any())
    #print(df.head())

    return df


def correlationMatrix(df):
    # get columns for correlation matrix
    # remove categorical/id data
    x = ["Unnamed: 0", "id", "item_id", "store_id", "weekday", "date"]
    # remove special cases
    # all 0 or all 1
    x.extend(["state_id:CA","cat_id:HOBBIES", "dept_id:HOBBIES_1", "item_store_last_day_sales_filled", "item_store_L7d_day_median_sales_filled", "item_store_L14d_day_median_sales_filled", "item_store_L21d_day_median_sales_filled", "item_store_L28d_day_median_sales_filled"])

    df_filtered = df.drop(columns=x)

    corrMat = df_filtered.corr()

    corrMat.to_csv("correlation.csv")

    strongPos = []
    mediumPos = []
    weakPos = []
    none = []
    weakNeg = []
    mediumNeg = []
    strongNeg = []

    for i in range(len(corrMat.columns)):
        for j in range(i + 1, len(corrMat.columns)):
            if (corrMat.iloc[i, j] > 0.7):
                strongPos.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            elif (corrMat.iloc[i, j] > 0.4):
                mediumPos.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            elif (corrMat.iloc[i, j] > 0.2):
                weakPos.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            elif (corrMat.iloc[i, j] > -0.2):
                none.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            elif (corrMat.iloc[i, j] > -0.4):
                weakNeg.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            elif (corrMat.iloc[i, j] > -0.7):
                mediumNeg.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")
            else:
                strongNeg.append(f"{corrMat.columns[i]} + {corrMat.columns[j]}: {corrMat.iloc[i, j]}")

    # print(strongPos + strongNeg)
    # print(mediumPos + mediumNeg)
    # print(weakPos + weakNeg)
    # print(none)

def graphs(df):
    df.plot(x="sell_price", y="sales", kind="scatter", marker="o", title="Sales vs Price")
    df.plot(x="wday", y="sales", kind="scatter", marker="o", title="Sales vs Weekday")
    df.plot(x="wday", y="sell_price", kind="scatter", marker="o", title="Price vs Weekday")

    df.plot(x="item_store_last_day_sales", y="sales", kind="scatter", marker="o", title="Sales vs Yesterday's Sales")
    df.plot(x="item_store_L7d_day_median_sales", y="sales", kind="scatter", marker="o", title="Sales vs Last Week's Sales")
    df.plot(x="item_store_L14d_day_median_sales", y="sales", kind="scatter", marker="o", title="Sales vs 2 Weeks Ago's Sales")
    df.plot(x="item_store_L21d_day_median_sales", y="sales", kind="scatter", marker="o", title="Sales vs 3 Weeks Ago's Sales")
    df.plot(x="item_store_L28d_day_median_sales", y="sales", kind="scatter", marker="o", title="Sales vs Last Month's Sales")

    df["log_sales"] = np.log(df["sales"] + 1)
    df["log_prices"] = np.log(df["sell_price"] + 1)

    df.plot(x="sell_price", y="log_sales", kind="scatter", marker="o", title="Log Sales vs Price")
    df.plot(x="item_store_last_day_sales", y="log_sales", kind="scatter", marker="o", title="Log Sales vs Yesterday's Sales")


    plt.show()
    eventSales = df[["event_name_1:Chanukah End","event_name_1:Christmas","event_name_1:Cinco De Mayo","event_name_1:ColumbusDay","event_name_1:Easter","event_name_1:Eid al-Fitr","event_name_1:EidAlAdha","event_name_1:Father's day","event_name_1:Halloween","event_name_1:IndependenceDay","event_name_1:LaborDay","event_name_1:LentStart","event_name_1:LentWeek2","event_name_1:MartinLutherKingDay","event_name_1:MemorialDay","event_name_1:Mother's day","event_name_1:NBAFinalsEnd","event_name_1:NBAFinalsStart","event_name_1:NewYear","event_name_1:OrthodoxChristmas","event_name_1:OrthodoxEaster","event_name_1:Pesach End","event_name_1:PresidentsDay","event_name_1:Purim End","event_name_1:Ramadan starts","event_name_1:StPatricksDay","event_name_1:SuperBowl","event_name_1:Thanksgiving","event_name_1:ValentinesDay","event_name_1:VeteransDay"]].multiply(df["sales"], axis=0).sum()
    eventSales.index = eventSales.index.str.replace("event_name_1:", "", regex=True)
    eventSales.plot(kind="pie", autopct="%1.1f%%", title="Sales per Event")
    plt.show()

def normalize(df):
    x = ["sales", "sell_price", "item_store_last_day_sales", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales", "item_store_L28d_day_median_sales"]
    df[x] = (df[x] - df[x].mean()) / df[x].std()


# example method usage
'''get_path()
test = [["name", "id"],["adam","1"],["eve","2"]]
csv_write("test.csv",test)
print(csv_read("test.csv"))'''

df = pd.read_csv("CS 578 datasets/merged_df.csv")
df.drop(columns=early_drop, inplace=True)
print("filling nulls")
df = fill_nulls(df)
print("encoding variables")
df = encode(df)
print("computing additional columns")
df = add_cols(df)
df.drop(columns=one_hot_encodeables,inplace=True)
df.to_csv("test.csv")
print("df prepared")

# uncomment this for analysis
'''correlationMatrix(df)

graphs(df)

normalize(df)'''

#train :2014-6-1 to 2016-1-31
#val: 2016-2-1 to 2016-4-30
#test: 2016-5-1 to 2016-5-22

df_train = df[(df["d"] >= 1220) & (df["d"] <= 1829)]
df_val = df[(df["d"] >= 1830) & (df["d"] <= 1919)]
df_test = df[(df["d"] >= 1920) & (df["d"] <= 1941)]

df_train = df_train.drop(columns = late_drop)
df_val = df_val.drop(columns = late_drop)
df_test = df_test.drop(columns = late_drop)



df_train.to_csv("training.csv")
df_val.to_csv("validation.csv")
df_test.to_csv("testing.csv")
#print(df['sell_price'].isnull().sum())
#print(df.groupby(["d","sell_price"]).count())