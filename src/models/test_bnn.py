import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
from scipy.stats import norm
from scipy.special import erf

from tensorflow import keras

from keras import layers

import pandas as pd

import matplotlib.pyplot as plt

tfd = tfp.distributions

gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# .\tf-env-1\Scripts\Activate.ps1

# conda activate tf210

num_epochs = 200

hidden_units = [128, 64, 32]
learning_rate = 0.001

scale = 5

def prior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    prior_model = keras.Sequential(
        [
            tfp.layers.DistributionLambda(
                lambda t: tfp.distributions.MultivariateNormalDiag(
                    loc=tf.ones(n) * 0.1, scale_diag=tf.ones(n) * scale
                )
            )
        ]
    )
    return prior_model

def posterior(kernel_size, bias_size, dtype=None):
    n = kernel_size + bias_size
    return tf.keras.Sequential([
        tfp.layers.VariableLayer(2 * n, dtype=dtype),
        tfp.layers.DistributionLambda(lambda t: tfd.MultivariateNormalDiag(
            loc=t[..., :n],
            scale_diag=1e-1 + tf.nn.softplus(tf.clip_by_value(t[..., n:], -10.0, 5.0))
        ))
    ])

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


df = pd.read_csv("training.csv")
df = df.dropna()

x = ["sales", "sell_price", "item_store_last_day_sales", "item_store_L7d_day_median_sales", "item_store_L14d_day_median_sales", "item_store_L21d_day_median_sales", "item_store_L28d_day_median_sales"]
mean1 = df[x].mean()
std1 = df[x].std()
df[x] = (df[x] - mean1) / std1

sales_mean = mean1.get("sales")
sales_std = std1.get("sales")

print("mean and std: ", sales_mean, sales_std)

train_size = len(df)

drop_col = ["Unnamed: 0.1", 'd', 'wm_yr_wk',  "sales", "Unnamed: 0", "id", "item_id", "weekday", "date", "state_id:CA", "cat_id:HOBBIES", "month"]

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

df_valid = pd.read_csv("validation.csv")

df_valid = df_valid.dropna()

df_valid[x] = (df_valid[x] - mean1) / std1

y_valid = np.array(df_valid["sales"]).reshape(-1, 1)

df_test = pd.read_csv("testing.csv")

df_test = df_test.dropna()

true_sales = df_test["sales"].values

print("minimum ", np.min(df_test["sales"]))

df_test[x] = (df_test[x] - mean1) / std1

print("minimum2 ", np.min((df_test[x] * std1 + mean1)["sales"]))

y_test = df_test["sales"].values

df["day_scaled"] = (df["d"] - df["d"].min()) / (df["d"].max() - df["d"].min())
df["day_sin"] = np.sin(2 * np.pi * df["day_scaled"])
df["day_cos"] = np.cos(2 * np.pi * df["day_scaled"])
df["day_of_week_sin"] = np.sin(2 * np.pi * df["wday"] / 7)
df["day_of_week_cos"] = np.cos(2 * np.pi * df["wday"] / 7)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

df_valid["day_scaled"] = (df_valid["d"] - df_valid["d"].min()) / (df_valid["d"].max() - df_valid["d"].min())
df_valid["day_sin"] = np.sin(2 * np.pi * df_valid["day_scaled"])
df_valid["day_cos"] = np.cos(2 * np.pi * df_valid["day_scaled"])
df_valid["day_of_week_sin"] = np.sin(2 * np.pi * df_valid["wday"] / 7)
df_valid["day_of_week_cos"] = np.cos(2 * np.pi * df_valid["wday"] / 7)
df_valid["month_sin"] = np.sin(2 * np.pi * df_valid["month"] / 12)
df_valid["month_cos"] = np.cos(2 * np.pi * df_valid["month"] / 12)

df_test["day_scaled"] = (df_test["d"] - df_test["d"].min()) / (df_test["d"].max() - df_test["d"].min())
df_test["day_sin"] = np.sin(2 * np.pi * df_test["day_scaled"])
df_test["day_cos"] = np.cos(2 * np.pi * df_test["day_scaled"])
df_test["day_of_week_sin"] = np.sin(2 * np.pi * df_test["wday"] / 7)
df_test["day_of_week_cos"] = np.cos(2 * np.pi * df_test["wday"] / 7)
df_test["month_sin"] = np.sin(2 * np.pi * df_test["month"] / 12)
df_test["month_cos"] = np.cos(2 * np.pi * df_test["month"] / 12)

FEATURE_NAMES.extend(["day_scaled", "day_sin", "day_cos", "day_of_week_sin", "day_of_week_cos", "month_sin", "month_cos"])

X_train = df[FEATURE_NAMES].values.astype(np.float32)
X_valid = df_valid[FEATURE_NAMES].values.astype(np.float32)
X_test = df_test[FEATURE_NAMES].values.astype(np.float32)

unique_items = df["id"].unique()
item_to_index = {item: i for i, item in enumerate(unique_items)}
df["item_idx"] = df["id"].map(item_to_index)
df_valid["item_idx"] = df_valid["id"].map(item_to_index)
df_test["item_idx"] = df_test["id"].map(item_to_index)

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

def compute_qce(y_true, y_pred_mean, y_pred_std, num_bins=10, alpha=1.0, z_thresh=3):
    # Remove outliers based on z-score
    z_scores = (y_true - y_pred_mean) / y_pred_std
    inlier_mask = np.abs(z_scores) < z_thresh

    # Filter arrays to exclude outliers
    y_true = y_true[inlier_mask]
    y_pred_mean = y_pred_mean[inlier_mask]
    y_pred_std = y_pred_std[inlier_mask]

    lower = y_pred_mean - alpha * y_pred_std
    upper = y_pred_mean + alpha * y_pred_std
    bins = np.linspace(y_pred_std.min(), y_pred_std.max(), num_bins + 1)
    bin_indices = np.digitize(y_pred_std, bins) - 1
    qce = 0.0
    total = len(y_true)
    expected_coverage = 2 * (0.5 * (1 + erf(alpha / np.sqrt(2))))
    for i in range(num_bins):
        mask = bin_indices == i
        if np.sum(mask) == 0:
            continue
        quantile_coverage = ((y_true[mask] >= lower[mask]) & (y_true[mask] <= upper[mask])).mean()
        qce += np.abs(quantile_coverage - expected_coverage) * np.sum(mask) / total
    return qce

def calibration_curve(y_true, y_pred_mean, y_pred_std, num_points=10):
    """
    Plots calibration curve for regression uncertainty.
    """
    quantiles = np.linspace(0.05, 0.95, num_points)
    nominal_coverage = []
    empirical_coverage = []

    for q in quantiles:
        alpha = norm.ppf(0.5 + q/2)  # Corresponding z-score for central interval
        lower = y_pred_mean - alpha * y_pred_std
        upper = y_pred_mean + alpha * y_pred_std
        # Empirical coverage: fraction of true values within the interval
        coverage = ((y_true >= lower) & (y_true <= upper)).mean()
        nominal_coverage.append(q)
        empirical_coverage.append(coverage)

    plt.figure(figsize=(6,6))
    plt.plot(nominal_coverage, empirical_coverage, marker='o', label='Model')
    plt.plot([0,1], [0,1], 'k--', label='Perfect Calibration')
    plt.xlabel('Nominal Confidence Level')
    plt.ylabel('Empirical Coverage')
    plt.title('Calibration Curve (Coverage Plot)')
    plt.legend()
    plt.grid(True)
    plt.show()


def scheduler(epoch, lr):
    if epoch < 5:
        return lr + 0.0001
    return lr

def run_experiment(model, X_test, Y_test):
    samples = np.stack([
        model(X_test, training=True).sample().numpy()
        for _ in range(50)
    ], axis=0)

    test_std = samples.std(axis=0)

    samples_unnorm = samples * sales_std + sales_mean

    test_std_unnorm = np.std(samples_unnorm, axis=0)  

    test_confidence = 1 / (1 + test_std)

    y_pred_test = samples.mean(axis=0)

    y_pred_test_unnorm = samples_unnorm.mean(axis=0)
    y_test_unnorm = true_sales.reshape(-1)

    # train_rmse = np.sqrt(np.mean((y_pred_train_unnorm - y_train_unnorm) ** 2))
    test_rmse1 = np.sqrt(np.mean((y_pred_test.reshape(-1) - y_test.reshape(-1)) ** 2))

    test_rmse = np.sqrt(np.mean((y_pred_test_unnorm.ravel() - y_test_unnorm.ravel()) ** 2))

    # print(f"Unnormalized Train RMSE: {round(train_rmse, 3)}")
    print(f"Normalized Test RMSE: {round(test_rmse1, 3)}")

    print(f"Unnormalized Test RMSE: {round(test_rmse, 3)}")

    # print(f"Train ECE: {round(expected_calibration_error(train_confidence, y_pred_train_unnorm, y_train_unnorm), 3)}")
    print(f"Test ECE: {round(expected_calibration_error(test_confidence, y_pred_test_unnorm, y_test_unnorm), 3)}")

    print(f"Test QCE: {round(compute_qce(y_test_unnorm.reshape(-1), y_pred_test_unnorm.reshape(-1), test_std_unnorm.reshape(-1)), 3)}")

    print(np.mean(test_std_unnorm), np.median(test_std_unnorm), np.max(test_std_unnorm))

    x = np.arange(len(Y_test))  # x-axis = sample index

    df_plot = pd.DataFrame({
        "day": df_test["d"].values.ravel(),         # Day index
        "y_true": Y_test.ravel(),           # Ground truth values
        "y_pred": y_pred_test.ravel(),           # Predicted values
        "y_std": test_std.ravel()
    })

    daily_avg = df_plot.groupby("day").agg({
        "y_true": "mean",
        "y_pred": "mean",
        "y_std": "mean"  # or "std" if you want std of stds (less common)
    })

    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg.index, daily_avg["y_true"], label="Actual Daily Avg")
    plt.plot(daily_avg.index, daily_avg["y_pred"], label="Predicted Daily Avg", linestyle="--")
    plt.fill_between(
        daily_avg.index,
        daily_avg["y_pred"] - daily_avg["y_std"],
        daily_avg["y_pred"] + daily_avg["y_std"],
        color='orange', alpha=0.3, label='±1 Stddev'
    )
    plt.xlabel("Day")
    plt.ylabel("Normalized Average Value")
    plt.title("Normalized Daily Average: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    df_plot_unnorm = pd.DataFrame({
        "day": df_test["d"].values.ravel(),         # Day index
        "y_true": y_test_unnorm.ravel(),           # Ground truth values
        "y_pred": y_pred_test_unnorm.ravel(),
        "y_std": test_std_unnorm.ravel()
    })

    daily_avg_unnorm = df_plot_unnorm.groupby("day").agg({
        "y_true": "mean",
        "y_pred": "mean",
        "y_std": "mean"  # or "std" if you want std of stds (less common)
    })

    plt.figure(figsize=(12, 6))
    plt.plot(daily_avg_unnorm.index, daily_avg_unnorm["y_true"], label="Actual Daily Avg")
    plt.plot(daily_avg_unnorm.index, daily_avg_unnorm["y_pred"], label="Predicted Daily Avg", linestyle="--")
    plt.fill_between(
        daily_avg_unnorm.index,
        daily_avg_unnorm["y_pred"] - daily_avg_unnorm["y_std"],
        daily_avg_unnorm["y_pred"] + daily_avg_unnorm["y_std"],
        color='orange', alpha=0.3, label='±1 Stddev'
    )
    plt.xlabel("Day")
    plt.ylabel("Unnormalized Average Value")
    plt.title("Unnormalized Daily Average: Actual vs Predicted")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    x = np.arange(500)  # x-axis = sample index

    plt.figure(figsize=(10, 5))
    plt.plot(x, Y_test[:500], label='Actual', linewidth=2)
    plt.plot(x, y_pred_test[:500], label='Predicted', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(x, y_test_unnorm[:500], label='Actual', linewidth=2)
    plt.plot(x, y_pred_test_unnorm[:500], label='Predicted', linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Actual vs Predicted')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    calibration_curve(y_test.reshape(-1), y_pred_test.reshape(-1), test_std.reshape(-1))

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
            kl_weight=1e-6 / (train_size),
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
def negative_loglikelihood(targets, estimated_distribution, penalty_weight=10.0):
    nll = -estimated_distribution.log_prob(targets)
 
    mu_norm  = estimated_distribution.mean()
    
    mu_unnorm = mu_norm * sales_std + sales_mean

    penalty = tf.nn.relu(-mu_unnorm)
    penalty_term = penalty_weight * penalty

    return nll

bnn_model = create_bnn_model(train_size)

bnn_model.load_weights("128,64,32,rms,sigmoid_newnll6.h5")

run_experiment(bnn_model, X_test_dict, y_test)
