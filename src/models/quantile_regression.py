import csv
import os
import torch
from torch import nn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

percentile = .1

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


df = pd.read_csv("training.csv")
#print(df.dtypes)
print(df.shape)

features = ["wday", "month", "year", "snap_CA", "snap_TX", "snap_WI", 
        "sell_price", "event_name_1_filled", "event_type_1_filled", "sell_price_filled", "dept_id:HOBBIES_1", 
        "cat_id:HOBBIES", "state_id:CA", "event_name_1:Chanukah End", "event_name_1:Christmas", "event_name_1:Cinco De Mayo", 
        "event_name_1:ColumbusDay", "event_name_1:Easter", "event_name_1:Eid al-Fitr", "event_name_1:EidAlAdha", 
        "event_name_1:Father's day", "event_name_1:Halloween", "event_name_1:IndependenceDay", "event_name_1:LaborDay", 
        "event_name_1:LentStart", "event_name_1:LentWeek2", "event_name_1:MartinLutherKingDay", "event_name_1:MemorialDay", 
        "event_name_1:Mother's day", "event_name_1:NBAFinalsEnd", "event_name_1:NBAFinalsStart", "event_name_1:NewYear", 
        "event_name_1:OrthodoxChristmas", "event_name_1:OrthodoxEaster", "event_name_1:Pesach End", "event_name_1:PresidentsDay", 
        "event_name_1:Purim End", "event_name_1:Ramadan starts", "event_name_1:StPatricksDay", "event_name_1:SuperBowl", 
        "event_name_1:Thanksgiving", "event_name_1:ValentinesDay", "event_name_1:VeteransDay", "event_name_1:no_event_name_1", 
        "event_type_1:Cultural", "event_type_1:National", "event_type_1:Religious", "event_type_1:Sporting", 
        "event_type_1:no_event_type_1", "last", "item_store_last_day_sales", "item_store_last_day_sales_filled", 
        "item_store_L7d_day_median_sales_filled", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales_filled", 
        "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales_filled", "item_store_L21d_day_median_sales", 
        "item_store_L28d_day_median_sales_filled", "item_store_L28d_day_median_sales"]
discarded = ["d", "sales", "date", "wm_yr_wk", "weekday"]

class quantile_regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(len(features), 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, 1)
        )
        

    def forward(self, x):
        return self.model(x)




# takes in dataframe for data
def train(model, optimizer, data, loss_funct, batches = 1000, batch_size = 2000):

    for epoch in range(2):
        shuffled = data.sample(frac = 1)
        x = torch.from_numpy(shuffled[features].values).float()
        y = torch.from_numpy(shuffled["sales"].values).float()
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
            running_loss += loss.item()
            if i % 10 == 9:    # print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss/10 :.5f}')
                running_loss = 0.0

def evaluate(model, data, loss_funct):
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data["sales"].values).float()
    pred = model(x)
    return loss_funct(pred, y)

    

'''model = quantile_regression()
optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

train(model, optimizer, df, quantile_loss())
torch.save(model.state_dict(), "quantile_regression"+str(int(percentile*100))+"percentile")

print(df[features].shape)

x = df[features]
y = df["sales"]'''
#print(x.dtypes)

models = []
for i in range(1, 10, 1):
    percentile = 0.1*i
    model = quantile_regression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.00001, momentum=0.9)

    
    train(model, optimizer, df, quantile_loss())
    models.append(model)
    torch.save(model.state_dict(), "quantile_regression"+str(int(percentile*100))+"percentile")
    print("saved percentile regression " + str(percentile))

for i in range(1, 10, 1):
    prin




    