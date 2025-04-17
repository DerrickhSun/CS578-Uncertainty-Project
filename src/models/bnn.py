import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

from tensorflow import keras

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disables GPU

from keras import layers

import pandas as pd

import matplotlib.pyplot as plt

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# tf.debugging.set_log_device_placement(True)

# .\tf-env-1\Scripts\Activate.ps1

# conda activate tf210

num_epochs = 200

tfd = tfp.distributions

df = pd.read_csv("training.csv")
df = df.dropna()

print("Number of entries: " + df.shape[0])

hidden_units = [100, 50]
learning_rate = 0.001

scale = 5

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.zeros(n), scale_diag=tf.ones(n) * scale
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return keras.Sequential([
        tfp.layers.VariableLayer(
            tfp.layers.MultivariateNormalTriL.params_size(n),
            dtype=dtype
        ),
        tfp.layers.MultivariateNormalTriL(n)
    ])


x = ["sales", "sell_price", "item_store_last_day_sales", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales", "item_store_L28d_day_median_sales"]
mean1 = df[x].mean()
std1 = df[x].std()
df[x] = (df[x] - mean1) / std1

sales_mean = mean1.get("sales")
sales_std = std1.get("sales")

train_size = len(df)

drop_col = ["Unnamed: 0.1", 'snap_CA_filled', 'snap_TX_filled', 'snap_WI_filled', 'date_filled', 'wm_yr_wk_filled', 'weekday_filled', 'wday_filled', 'month_filled', 'year_filled', 'd', 'wm_yr_wk',  "sales", "Unnamed: 0", "id", "item_id", "weekday", "date", "state_id:CA","cat_id:HOBBIES"]

df_dropped = df.drop(columns=drop_col)

FEATURE_NAMES = list(df_dropped)

y_train = np.array(df["sales"])
y_train = y_train.reshape(-1, 1)  # Reshape for TensorFlow

def create_model_inputs():
    inputs = {}
    for feature_name in FEATURE_NAMES:
        inputs[feature_name] = layers.Input(
            name=feature_name, shape=(1,), dtype=tf.float32
        )
    return inputs

def make_lstm_sequences(df, item_to_index, sequence_length=28):
    X_seq = []
    y = []
    item_ids = []

    grouped = df.groupby("id")

    for item_id, group in grouped:
        group = group.sort_values("d")  # sort by time
        features = group[FEATURE_NAMES].to_numpy()
        targets = group["sales"].to_numpy()

        for i in range(len(group) - sequence_length):
            X_seq.append(features[i:i+sequence_length])
            y.append(targets[i + sequence_length])
            item_ids.append(item_to_index.get(item_id, 0))  # use 0 as fallback index

    return (
        np.array(X_seq).astype(np.float32),
        np.array(y).reshape(-1, 1).astype(np.float32),
        np.array(item_ids).reshape(-1, 1).astype(np.int32)
    )

def extract_static_for_sequences(df, sequence_length):
    grouped = df.groupby("id")
    statics = []

    for _, group in grouped:
        group = group.sort_values("d")
        for i in range(len(group) - sequence_length):
            statics.append(group.iloc[i + sequence_length][FEATURE_NAMES].to_numpy())

    return np.array(statics).astype(np.float32)

df_valid = pd.read_csv("validation.csv")

df_valid = df_valid.dropna()

df_valid[x] = (df_valid[x] - mean1) / std1

y_valid = np.array(df_valid["sales"]).reshape(-1, 1)

df_test = pd.read_csv("testing.csv")

df_test = df_valid.dropna()

df_test[x] = (df_test[x] - mean1) / std1

y_test = np.array(df_test["sales"])

# df["day_scaled"] = (df["d"] - df["d"].min()) / (df["d"].max() - df["d"].min())
# df["day_sin"] = np.sin(2 * np.pi * df["day_scaled"])
# df["day_cos"] = np.cos(2 * np.pi * df["day_scaled"])
# df["day_of_week_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
# df["day_of_week_cos"] = np.cos(2 * np.pi * df["wday"] / 7)
# df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
# df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

# df_valid["day_scaled"] = (df_valid["d"] - df_valid["d"].min()) / (df_valid["d"].max() - df_valid["d"].min())
# df_valid["day_sin"] = np.sin(2 * np.pi * df_valid["day_scaled"])
# df_valid["day_cos"] = np.cos(2 * np.pi * df_valid["day_scaled"])
# df_valid["day_of_week_sin"] = np.sin(2 * np.pi * df_valid["wday"] / 7)
# df_valid["day_of_week_cos"] = np.cos(2 * np.pi * df_valid["wday"] / 7)
# df_valid["month_sin"] = np.sin(2 * np.pi * df_valid["month"] / 12)
# df_valid["month_cos"] = np.cos(2 * np.pi * df_valid["month"] / 12)

# df_test["day_scaled"] = (df_test["d"] - df_test["d"].min()) / (df_test["d"].max() - df_test["d"].min())
# df_test["day_sin"] = np.sin(2 * np.pi * df_test["day_scaled"])
# df_test["day_cos"] = np.cos(2 * np.pi * df_test["day_scaled"])
# df_test["day_of_week_sin"] = np.sin(2 * np.pi * df_test["wday"] / 7)
# df_test["day_of_week_cos"] = np.cos(2 * np.pi * df_test["wday"] / 7)
# df_test["month_sin"] = np.sin(2 * np.pi * df_test["month"] / 12)
# df_test["month_cos"] = np.cos(2 * np.pi * df_test["month"] / 12)

# FEATURE_NAMES.extend(["day_scaled", "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"])

X_train = df[FEATURE_NAMES].values.astype(np.float32)
X_valid = df_valid[FEATURE_NAMES].values.astype(np.float32)
X_test = df_test[FEATURE_NAMES].values.astype(np.float32)

unique_items = df["id"].unique()
item_to_index = {item: i for i, item in enumerate(unique_items)}
df["item_idx"] = df["id"].map(item_to_index)
df_valid["item_idx"] = df_valid["id"].map(item_to_index)
df_test["item_idx"] = df_test["id"].map(item_to_index)

# X_seq, y_seq, item_id_seq = make_lstm_sequences(df, item_to_index, sequence_length=28)
# X_static = extract_static_for_sequences(df, 28)

# X_seq_val, y_seq_val, item_id_seq_val = make_lstm_sequences(df_valid, item_to_index, sequence_length=28)
# X_static_val = extract_static_for_sequences(df_valid, 28)

# X_seq_test, y_seq_test, item_id_seq_test = make_lstm_sequences(df_test, item_to_index, sequence_length=28)
# X_static_test = extract_static_for_sequences(df_test, 28)

X_train_dict = {
    "features": X_train,
    "item_id": df["item_idx"].values.reshape(-1, 1).astype(np.int32)
}

X_valid_dict = {
    "features": X_valid,
    "item_id": df_valid["item_idx"].values.reshape(-1, 1).astype(np.int32)
}

X_test_dict = {
    "features": X_test,
    "item_id": df_test["item_idx"].values.reshape(-1, 1).astype(np.int32)
}

num_unique_items = len(item_to_index)

def expected_calibration_error(confidence, y_true, y_pred):
    confidence = np.squeeze(np.array(confidence))
    y_true = np.squeeze(np.array(y_true))
    y_pred = np.squeeze(np.array(y_pred))

    bins = np.linspace(0, 1, 10 + 1)
    bin_indices = np.digitize(confidence, bins) - 1 

    error = np.abs(y_pred - y_true)

    error = np.abs(y_pred - y_true)

    error = error / (np.max(np.abs(y_true)) + 1e-8) 

    ece = 0.0
    for i in range(10):
        bin_mask = bin_indices == i
        bin_size = np.sum(bin_mask)

        if bin_size > 0:
            bin_confidence = np.mean(confidence[bin_mask])
            bin_error = np.mean(error[bin_mask])
            ece += np.abs(bin_confidence - (1 - bin_error)) * (bin_size / len(y_true))

    return ece

def scheduler(epoch, lr):
    if epoch < 5:
        return lr + 0.0001
    return lr


def run_experiment(model, loss, X_train, Y_train, X_valid, Y_valid, X_test, Y_test):

    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=learning_rate),
        loss=loss,
        metrics=[keras.metrics.RootMeanSquaredError()],

    )

    callbacks = [
        keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        keras.callbacks.LearningRateScheduler(scheduler)
    ]

    print("Start training the model...")
    model.fit(X_train, Y_train, epochs=num_epochs, batch_size=256, validation_data=(X_valid, Y_valid), callbacks=callbacks)
    bnn_model.save_weights("100,50,rms,sigmoid.h5")    
    print("Model training finished.")
    
    _, rmse = model.evaluate(X_train, Y_train, verbose=0)
    print(f"Train RMSE: {round(rmse, 3)}")

    print("Evaluating model performance...")
    _, rmse = model.evaluate(X_test, Y_test, verbose=0) 
    print(f"Test RMSE: {round(rmse, 3)}")

    # Unnormalized RMSE
    # predictions_train = model(X_train, training=False)
    predictions_test = model(X_test, training=False)

    # train_std = predictions_train.stddev().numpy()
    test_std = predictions_test.stddev().numpy()

    # train_confidence = 1 / (1 + train_std)
    test_confidence = 1 / (1 + test_std)

    # y_pred_train = predictions_train.mean().numpy() # shape (n, 1)
    y_pred_test = predictions_test.mean().numpy() # shape (n, 1)

    # y_pred_train_unnorm = y_pred_train * sales_std + sales_mean
    # y_train_unnorm = Y_train * sales_std + sales_mean

    y_pred_test_unnorm = y_pred_test * sales_std + sales_mean
    y_test_unnorm = Y_test * sales_std + sales_mean

    # y_pred_train_unnorm = y_pred_train_unnorm.reshape(-1)
    # y_train_unnorm = y_train_unnorm.reshape(-1)

    y_pred_test_unnorm = y_pred_test_unnorm.reshape(-1)
    y_test_unnorm = y_test_unnorm.reshape(-1)

    # y_pred_test_unnorm = y_pred_test_unnorm.reshape(-1)
    # y_test_unnorm = y_test_unnorm.reshape(-1)

    # train_rmse = np.sqrt(np.mean((y_pred_train_unnorm - y_train_unnorm) ** 2))
    test_rmse = np.sqrt(np.mean((y_pred_test_unnorm - y_test_unnorm) ** 2))

    # print(f"Unnormalized Train RMSE: {round(train_rmse, 3)}")
    print(f"Unnormalized Test RMSE: {round(test_rmse, 3)}")

    # print(f"Train ECE: {round(expected_calibration_error(train_confidence, y_pred_train_unnorm, y_train_unnorm), 3)}")
    print(f"Test ECE: {round(expected_calibration_error(test_confidence, y_pred_test_unnorm, y_test_unnorm), 3)}")

    x = np.arange(len(Y_test))  # x-axis = sample index

    df_plot = pd.DataFrame({
        "day": df_test["d"].values.ravel(),         # Day index
        "y_true": Y_test.ravel(),           # Ground truth values
        "y_pred": y_pred_test.ravel()            # Predicted values
    })

    daily_avg = df_plot.groupby("day").mean()

    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg.index, daily_avg["y_true"], label="Actual Daily Avg")
    plt.plot(daily_avg.index, daily_avg["y_pred"], label="Predicted Daily Avg", linestyle="--")
    plt.xlabel("Day")
    plt.ylabel("Average Value")
    plt.title("Daily Average: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    x = np.arange(len(Y_test))  # x-axis = sample index

    plt.figure(figsize=(10, 5))
    plt.plot(x, Y_test, label='Actual', linewidth=2)
    plt.plot(x, y_pred_test, label='Predicted', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def create_bnn_model(train_size):
    # Input for dense numerical features (already normalized)
    dense_input = keras.Input(shape=(len(FEATURE_NAMES),), name="features")

    # Input for item index (integer ID)
    item_input = keras.Input(shape=(1,), dtype=tf.int32, name="item_id")
    item_emb = keras.layers.Embedding(
        input_dim=num_unique_items,
        output_dim=min(50, num_unique_items // 2)
    )(item_input)
    item_emb = tf.keras.layers.GlobalAveragePooling1D()(item_emb)

    # Combine embedding + dense features
    combined = keras.layers.Concatenate()([dense_input, item_emb])
    features = layers.BatchNormalization()(combined)

    for units in hidden_units:
        features = tfp.layers.DenseVariational(
            units=units,
            make_prior_fn=prior,
            make_posterior_fn=posterior,
            kl_weight=0.000001 / (train_size),
            activation=keras.activations.sigmoid,
        )(features)

    distribution_params = layers.Dense(units=2)(features)
    outputs = tfp.layers.DistributionLambda(
        lambda t: tfd.Independent(
            tfd.Normal(
                loc=t[..., :1],
                scale=1e-2 + tf.nn.softplus(tf.clip_by_value(t[..., 1:], -10, 5))
            ),
            reinterpreted_batch_ndims=1
        )
    )(distribution_params)

    # outputs = tfp.layers.IndependentNormal(1)(distribution_params)
    
    model = keras.Model(inputs={"features": dense_input, "item_id": item_input}, outputs=outputs)

    return model

@tf.function
def negative_loglikelihood(targets, estimated_distribution):
    return -estimated_distribution.log_prob(targets)

with tf.device("/GPU:0"):
    bnn_model = create_bnn_model(train_size)

run_experiment(bnn_model, negative_loglikelihood, X_train_dict, y_train, X_valid_dict, y_valid, X_test_dict, y_test)