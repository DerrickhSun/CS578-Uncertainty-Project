import csv
import os
import torch
from torch import nn
import time

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

percentile = .1
losses = []

'''def quantile_loss(output, target):
    diff = torch.sub(target - output)
    diff1 = torch.mul(diff, percentile)
    diff2 = torch.mul(diff, 1 - percentile)
    return torch.mean(torch.maximum(diff1, diff2))'''
class quantile_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        # Implementation of custom loss calculation
        diff = torch.sub(y_true, y_pred)
        diff1 = torch.mul(diff, percentile)
        diff2 = torch.mul(diff, percentile - 1)
        return torch.mean(torch.maximum(diff1, diff2))
    
class multi_quantile_loss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, y_pred, y_true):
        sum = 0
        for i in range(1, 19, 1):
            percentile = i*0.05
            #print(y_pred.shape)
            #print(y_true.shape)
            diff = torch.sub(y_true, y_pred[:,i-1])
            #print(diff)
            #print("printing")
            #print(y_true)
            #print(y_pred)
            #print(diff)
            diff1 = torch.mul(diff, percentile)
            diff2 = torch.mul(diff, percentile - 1)
            sum += torch.mean(torch.maximum(diff1, diff2))
        return sum/19

df = pd.read_csv("data/training.csv")
#print(df.dtypes)

features = ["item_id", "wday", "month", "year", "snap_CA", "sell_price", "event_name_1_filled", "event_type_1_filled", "sell_price_filled", 
            "dept_id:HOBBIES_1", "store_id:CA_1","store_id:CA_2","store_id:CA_3","store_id:CA_4","cat_id:HOBBIES", "state_id:CA", "event_name_1:Chanukah End", "event_name_1:Christmas", 
            "event_name_1:Cinco De Mayo", "event_name_1:ColumbusDay", "event_name_1:Easter", "event_name_1:Eid al-Fitr", "event_name_1:EidAlAdha", 
            "event_name_1:Father's day", "event_name_1:Halloween", "event_name_1:IndependenceDay", "event_name_1:LaborDay", 
            "event_name_1:LentStart", "event_name_1:LentWeek2", "event_name_1:MartinLutherKingDay", "event_name_1:MemorialDay", 
            "event_name_1:Mother's day", "event_name_1:NBAFinalsEnd", "event_name_1:NBAFinalsStart", "event_name_1:NewYear", 
            "event_name_1:OrthodoxChristmas", "event_name_1:OrthodoxEaster", "event_name_1:Pesach End", "event_name_1:PresidentsDay", 
            "event_name_1:Purim End", "event_name_1:Ramadan starts", "event_name_1:StPatricksDay", "event_name_1:SuperBowl", 
            "event_name_1:Thanksgiving", "event_name_1:ValentinesDay", "event_name_1:VeteransDay", "event_name_1:no_event_name_1", 
            "event_type_1:Cultural", "event_type_1:National", "event_type_1:Religious", "event_type_1:Sporting", 
            "event_type_1:no_event_type_1", "item_store_last_day_sales", "item_store_last_day_sales_filled", 
            "item_store_L7d_day_median_sales_filled", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales_filled", 
            "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales_filled", "item_store_L21d_day_median_sales", 
            "item_store_L28d_day_median_sales_filled", "item_store_L28d_day_median_sales"]
discarded = ["sales", "d", "Unnamed: 0", "Unnamed: 0.1", "id", "state_id:CA", "cat_id:HOBBIES"]
features = [col for col in df.columns if col not in discarded]
df["log_sales"] = np.log(df[['sales']] + 1)
target = "sales"
#features = ["item_id", "wday", "month", "year", "snap_CA", "sell_price", "sell_price_filled"]
#print(len(features))
#discarded = ["sales"]
#print(features)

class quantile_regression(nn.Module):
    def __init__(self, n = 500, multi=True):
        super().__init__()

        n_categories = df["item_id"].nunique()
        embed_dim = min(100, (n_categories))

        self.item_id_embedding = nn.Embedding(n_categories, embed_dim)
        output = 1
        if multi: output=19
        self.nn = nn.Sequential(
            nn.Linear(embed_dim + len(features) - 1, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, n),
            nn.ReLU(),
            nn.Linear(n, output)
        )
        

    def forward(self, x):
        embed = self.item_id_embedding(x[:,0].int())
        x = torch.cat((embed, x[:,1:]), dim = 1)
        return self.nn(x)


# takes in dataframe for data
def train(model, optimizer, data, loss_funct, batches = 1000, batch_size = 5, multi = True, show = True):
    losses = []
    start = time.time()
    epoch = 0
    stop = 50
    while epoch < stop and epoch < 2500:
        shuffled = data.sample(frac = 1)
        x = torch.from_numpy(shuffled[features].values).float()
        y = torch.from_numpy(shuffled[target].values).float()
        running_loss = 0.0
        for i in range(min(batches, int(data.shape[0]/batch_size))):#range(data.shape[0]):
            inputs = x[batch_size*i : batch_size*(i+1)]
            labels = y[batch_size*i : batch_size*(i+1)]

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_funct(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            if i % 1000 == 999 and show:    # print every 10 mini-batches
                current = time.time()
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10 :.5f}, {current-start :.5f} seconds')
                if multi: 
                    print("pred vs actual examples:", outputs[:5,5].detach(), labels[:5])
                    print("quantiles example:", outputs[:1].detach())
                else:
                    print("pred " + str(percentile) + " vs actual examples:", outputs[:5].detach(), labels[:5])
                losses.append(running_loss)
                running_loss = 0.0
                start = current

        if multi:
            test = model(x[:1])
            valid = True
            for i in range(18):
                if test[0][i+1] < test[0][i]:
                    valid = False
                    break
            print("valid:",valid)
            if not valid:
                stop+=1
        epoch+=1

def evaluate(model, data, loss_funct):
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data["sales"].values).float()
    pred = model(x)
    return loss_funct(pred, y)

def evaluate_MSE(model, data):
    #print(data[features+["sales"]][:5])
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data["sales"].values).float()
    pred = model(x)[:,9]
    #print("examples:", pred[:5], y[:5])
    
    return torch.mean(torch.square(torch.subtract(pred, y)))
    

def evaluate_model(model, data):
    print("loss:", evaluate(model, data, multi_quantile_loss()))
    print("MSE:", evaluate_MSE(model, data))


# trains a model
if __name__ == "__main__":
    '''file1 = open('output_test1.txt', 'w')

    model = quantile_regression(n = 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.000000001)
    train(model, optimizer, df, multi_quantile_loss())

    torch.save(model.state_dict(), "nn_1000_5p")
    np.save("loss_array.npy", losses)


    df_valid = pd.read_csv("data/validation.csv")
    print("validation loss:", evaluate(model, df_valid, multi_quantile_loss()), file=file1)
    df_test = pd.read_csv("data/testing.csv")
    print("testing loss:", evaluate(model, df_test, multi_quantile_loss()), file=file1)

    df_train = pd.read_csv("data/training.csv")
    print("training MSE:", evaluate_MSE(model, df_train), file=file1)
    print("testing MSE:", evaluate_MSE(model, df_test), file=file1)


    target = "log_sales"
    file1 = open('output_test1_log.txt', 'w')

    model = quantile_regression(n = 1000)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00000001)
    train(model, optimizer, df, multi_quantile_loss())

    torch.save(model.state_dict(), "nn_1000_5p_log")
    np.save("loss_array_log.npy", losses)


    df_valid = pd.read_csv("data/validation.csv")
    print("validation loss:", evaluate(model, df_valid, multi_quantile_loss()), file=file1)
    df_test = pd.read_csv("data/testing.csv")
    print("testing loss:", evaluate(model, df_test, multi_quantile_loss()), file=file1)

    df_train = pd.read_csv("data/training.csv")
    print("training MSE:", evaluate_MSE(model, df_train), file=file1)
    print("testing MSE:", evaluate_MSE(model, df_test), file=file1)'''
    for i in range(5,100,5):
        percentile = i/100.0
        model = quantile_regression(n=1000,multi=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.00000001)
        train(model, optimizer, df, quantile_loss(), multi=False)

        torch.save(model.state_dict(), "nn_p"+str(i))
        np.save("loss_array_"+str(i)+".npy", losses)


