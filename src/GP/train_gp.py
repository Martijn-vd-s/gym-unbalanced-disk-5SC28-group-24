"""
Gaussian Process Regression pipeline for the Unbalanced Disk System.

This script handles loading raw continuous time-series data from an .npz file
and splits the processed data into training, validation, and test sets (60:20:20).

The data from the inverted pendulum is dynamic and time-dependent.
Gaussian Processes are fundamentally static regression models,
which expects static relationship between the input and output data.

Due to the dynamic nature of the inverted pendulum the data needs to be converted.
The model structure is parameterized using a NARX model since it is assumed that the data contains noise.
"""
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import helper_functions as hf


if __name__ == "__main__":
    data, keys = hf.load_npz_file()
    # print(f"Keys: {keys}")

    th_data = data["th"]  # Theta a.k.a. angular position (35000,)
    u_data = data["u"]  # Control input (35000,)

    na = 2  # Input lags
    nb = 8  # Output lags

    # X_narx = (34998, 4), Y_narx = (34998,)
    X_narx, Y_narx = hf.make_training_data(u_data, th_data, na=na, nb=nb)

    # Split the data into a train, validation and test set
    x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = (
        hf.train_test_val_split(X_narx, Y_narx)
    )  # x_train_raw = (0.6 * 34998, 4), y_train_raw = (0.6 * 34998,)

    # ------------- Normalize the data --------------------------------------
    x_train_scaled, x_val_scaled, x_test_scaled, x_mean, x_std = hf.normalize_data(
        x_train_raw, x_val_raw, x_test_raw
    )

    y_train_scaled, y_val_scaled, y_test_scaled, y_mean, y_std = hf.normalize_data(
        y_train_raw, y_val_raw, y_test_raw
    )

    np.savez("scaler_params.npz", x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std)
    # ---------------------------------------------------------------------------

    # Train the GP model
    sparse_model = hf.gaussian_process(x_train_scaled, y_train_scaled, num_inducing=50)

    # sparse_model = load_model()  
    CURRENT_GP_DIR = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(CURRENT_GP_DIR, "sparse_gp_model.pkl")
    print(save_path)
    hf.save_model(model=sparse_model, path=save_path)

    # ----------- Predict using trained model ---------------------------------------
    y_train_pred_scaled, _ = sparse_model.predict(x_train_scaled)  # (20998, 1)
    y_val_pred_scaled, _ = sparse_model.predict(x_val_scaled)
    y_test_pred_scaled, _ = sparse_model.predict(x_test_scaled)

    y_train_pred, y_val_pred, y_test_pred = hf.denormalize_data(
        y_train_pred_scaled, y_val_pred_scaled, y_test_pred_scaled, y_mean, y_std
    )

    y_train_pred = y_train_pred.flatten()  # (20998,)
    y_val_pred = y_val_pred.flatten()  # (20998,)
    y_test_pred = y_test_pred.flatten()  # (20998,)

    # ------------ Analyse performance ----------------------------------------------
    train_rms_rad = np.sqrt(np.mean((y_train_pred - y_train_raw) ** 2))
    val_rms_rad = np.sqrt(np.mean((y_val_pred - y_val_raw) ** 2))
    test_rms_rad = np.sqrt(np.mean((y_test_pred - y_test_raw) ** 2))

    summary_data = {
        "Dataset Split": ["Training Set", "Validation Set", "Test Set"],
        "RMSE [rad]": [train_rms_rad, val_rms_rad, test_rms_rad],
        "RMSE [deg]": [
            train_rms_rad / (2 * np.pi) * 360,
            val_rms_rad / (2 * np.pi) * 360,
            test_rms_rad / (2 * np.pi) * 360,
        ],
        "NRMSE [%]": [
            train_rms_rad / np.std(y_train_raw) * 100,
            val_rms_rad / y_val_raw.std() * 100,
            test_rms_rad / y_test_raw.std() * 100,
        ],
    }

    metrics_df = pd.DataFrame(summary_data)
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv("csv_files/model_training_results.csv", index=True)

    # ----------- Visualize the results -----------------------
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(y_train_pred - y_train_raw, label="Train residual", linewidth=2)
    ax.plot(y_val_pred - y_val_raw, label="Validation residual", linewidth=2)
    ax.plot(y_test_pred - y_test_raw, label="Test residual", linewidth=2)

    ax.set_title(
        "Model residual data",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Angular Position [rad]", fontsize=12)
    ax.grid()
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig("images/model_training_result.png", dpi=300, bbox_inches="tight")
    # plt.show()

