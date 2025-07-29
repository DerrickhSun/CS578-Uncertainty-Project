import torch
import pandas as pd
from quantile_regression import quantile_regression
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("data/training.csv")
discarded = ["sales", "d", "Unnamed: 0", "Unnamed: 0.1", "id", "state_id:CA", "cat_id:HOBBIES"]
features = [col for col in df.columns if col not in discarded]
df["log_sales"] = np.log(df[['sales']] + 1)

def evaluate(model, data, loss_funct):
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data["sales"].values).float()
    pred = model(x)
    return loss_funct(pred, y)

def evaluate_MSE(model, data):
    #print(data[features+["sales"]][:5])
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data["sales"].values).float()
    #pred = model(x)[:,5]'
    pred = model(x)[:,0]
    #print("examples:", pred[:5], y[:5])
    
    return torch.mean(torch.square(torch.subtract(pred, y)))

def calibration_error(model, data):
    accuracies = calibration_curve_direct(model, data)
    error = 0
    for i in range(1, 10, 1):
        error += abs(accuracies[i-1]-i/10)/9
    return error


def calibration_curve(model, data, target = "sales"):
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data[target].values).float()
    y_hat = model(x)

    accuracies = []
    for i in range(9):
        lower_bound = y_hat[:,8-i]
        upper_bound = y_hat[:,10+i]
        correct = torch.ge(y, lower_bound) & torch.le(y, upper_bound)
        acc = torch.mean(correct.float())
        accuracies.append(acc)
        #print(acc)
        #print([lower_bound[0], upper_bound[0]])
    return accuracies

# janky, meant to be used with many models, each handling a different percentile
def calibration_curve_direct(data, target = "sales"):
    x = torch.from_numpy(data[features].values).float()
    y = torch.from_numpy(data[target].values).float()

    accuracies = []
    for i in range(1, 10, 1):
        lower_bound_model = quantile_regression(n=1000, multi=False)
        lower_bound_model.load_state_dict(torch.load("nn_p"+str(50-i*5)))
        lower_bound = lower_bound_model(x)

        upper_bound_model = quantile_regression(n=1000, multi=False)
        upper_bound_model.load_state_dict(torch.load("nn_p"+str(50+i*5)))
        upper_bound = upper_bound_model(x)

        correct = torch.ge(y, lower_bound) & torch.le(y, upper_bound)
        acc = torch.mean(correct.float())
        accuracies.append(acc)
        #print(acc)
        print(str(i*10)+" conf: ", acc)
        print([lower_bound[0], upper_bound[0]])
    return accuracies

# Jitter function
def jitter(arr, amount=0.2):
    return arr + np.random.uniform(-amount, amount, size=len(arr))

model = quantile_regression(n=1000)
model.load_state_dict(torch.load('nn_1000_5p', weights_only=True))
#model = quantile_regression(n=1000, multi=False)
#model.load_state_dict(torch.load('nn_p50', weights_only=True))

#log_model = quantile_regression(n=1000)
#log_model.load_state_dict(torch.load('nn_1000_5p_log', weights_only=True))


df = pd.read_csv("data/testing.csv")
print(evaluate_MSE(model, df))
print(df[["sales"]].std())
print(df[["sales"]].max())
print(df[["sales"]].mean())
#print(calibration_error(model, df))
#df["log_sales"] = np.log(df[['sales']] + 1)
# plot of calibration curve
'''plt.figure(figsize=(6,6))
#plt.plot([num / 100 for num in range(10,100,10)], calibration_curve(model, df), linestyle='-', marker='o')
plt.plot([num / 100 for num in range(10,100,10)], calibration_curve_direct(df), linestyle='-', marker='o')
plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
# Add labels and title
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Separated Quantile Regression Accuracy vs Confidence')
# Show the plot
plt.legend()
plt.grid(True)
plt.show()'''


# plot of actual vs predicted sales
x = torch.from_numpy(df[features].values).float()
y = torch.from_numpy(df["sales"].values).float()
#pred = model(x)[:,9]
pred = model(x)[:,0]
plt.scatter(jitter(pred.detach().numpy()), jitter(y), s=.5)
# Add labels and title
plt.xlabel('Predicted Sales')
plt.ylabel('Actual Sales')
plt.title('Separated Quantile Regression Actual vs Predicted')
# Show the plot
plt.grid(True)
plt.show()


df["pred"] = pred.detach().numpy()
df["d"] = df["d"]-1919

# plot sales and predictions vs index
plt.figure(figsize=(15,6))
df["sales"][:500].plot(legend=True)
df["pred"][:500].plot(legend=True)
plt.xlabel('Index')
plt.ylabel('Sales')
plt.title('Separated Quantile Regression Sales vs Index')
# Show the plot
plt.grid(True)
plt.show()



# plot of avg sales and predictions over time
df_group = df[["d","sales", "pred"]].groupby("d").mean()
df_group["sales"].plot(legend=True)
df_group["pred"].plot(legend=True)
# Add labels and title
plt.xlabel('Day')
plt.ylabel('Avg Sales')
plt.title('Separated Quantile Regression Sales vs Day')
# Show the plot
plt.grid(True)
plt.show()

'''plt.plot([num / 100 for num in range(10,100,10)], calibration_curve(log_model, df, "log_sales"), color='red', linestyle='-', marker='o')
plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
# Add labels and title
plt.xlabel('Confidence')
plt.ylabel('Accuracy')
plt.title('Quantile Regression Accuracy vs Confidence')
# Show the plot
plt.grid(True)
plt.show()'''