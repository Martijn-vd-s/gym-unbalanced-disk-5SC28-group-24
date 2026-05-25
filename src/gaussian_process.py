"""
Gaussian Process Regression pipeline for the Unbalanced Disk System.

This script handles loading raw continuous time-series data from an .npz file
and splits the processed data into training, validation, and test sets (60:20:20).

The data from the inverted pendulum is dynamic and time-dependent.
Gaussian Processes are fundamentally static regression models,
which expects static relationship between the input and output data.
Due to the dynamic nature of the inverted pendulum the data needs to be converted using a NARX/NOE/nonlinear
state-space.
"""

import os
import GPy
import pickle
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

def load_npz_file(
    filename: str = "training-val-test-data.npz", folder: str = "disc-benchmark-files"
) -> np.lib.npyio.NpzFile:
    """
    Retreives the data from a npz file.
    The file should be in a specific sub-folder in the  folder.

    Parameters
    ----------
    filename: str
        The name of the file with the data.
    folder: str
        The subfolder of `gym-unbalanced-disk-5SC28-group-24` where the datafile is located.

    Returns
    -------
        The loaded npz file and the data keys
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(script_dir)

    data_file = os.path.join(parent_dir, folder, filename)
    data = np.load(data_file)

    try:
        keys = data.files
    except:
        keys = None

    return data, keys

def train_test_val_split(
    x_data, y_data, train_percentage: float = 0.6, val_percentage: float = 0.2
):
    """
    Slices the data into a training, validation and test set.

    Parameters
    ----------
    x_data:
        The input feature data
    y_data:
        The target or output data
    train_percentage : float
        The proportion of the dataset to include in the training split
    val_percentage : float
        The proportion of the dataset to include in the validation split

    Returns
    -------
        The x_train, y_train, x_val, y_val, x_test, y_test data subsets
    """

    if len(x_data) != len(y_data):
        raise ValueError(
            f"x_data and y_data do not have the same amount of samples."
            f"Got len(X) = {len(x_data)} and len(y) = {len(y_data)}"
        )

    total_samples = len(x_data)

    train_end = int(total_samples * train_percentage)
    val_end = int(total_samples * (train_percentage + val_percentage))

    x_train = x_data[:train_end]
    x_val = x_data[train_end:val_end]
    x_test = x_data[val_end:]

    y_train = y_data[:train_end]
    y_val = y_data[train_end:val_end]
    y_test = y_data[val_end:]

    return x_train, y_train, x_val, y_val, x_test, y_test

def make_training_data(x_data, y_data, na: int = 2, nb: int = 2):
    """
    Format raw time-series data into a Nonlinear AutoRegressive with Exogenous
    inputs (NARX) data structure for system identification.

    This function maps a continuous time-series dataset into a static regression
    matrix by constructing lagged feature vectors. Each row in the resulting feature
    matrix X represents the historical state of past inputs and past outputs over
    a finite time horizon, which serves to predict the corresponding current state
    target in Y.

    Parameters
    ----------
    x_data : np.ndarray
        Array of the system's inputs over time
    y_data : np.ndarray
        Array of the system's measured outputs over time
    na : int
        Specifies the number of past output lags to include in the state feature vector.
    nb : int
        Specifies the number of past input lags to include in the state feature vector.

    Returns
    -------
    Xdata : np.ndarray
        The regression feature matrix. (Nsamples,Nfeatures)
    Ydata : np.ndarray
        The ground-truth target vector a.k.a. true system state. (Nsamples)
    """
    Xdata = []
    Ydata = []
    # for loop that skips some data to account for the memory
    for k in range(max(na, nb), len(x_data)):
        # Slices the data to get a window of past input data (x_data[k - nb : k]) and
        # past output data (y_data[k - na : k]), it excludes the current time step k
        x_lags = x_data[k - nb : k]
        y_lags = y_data[k - na : k]

        # Fuses x_slice and y_slice into a single 1D array
        Xdata.append(np.concatenate([x_lags, y_lags]))
        Ydata.append(y_data[k])
    return np.array(Xdata), np.array(Ydata)

def inducing_point_search(x_data, y_data):
    nr_points = [10, 20, 30, 40, 50, 75, 100, 150]

    results = {}

    for points in nr_points:
        model = gaussian_process(x_data, y_data, num_inducing=points, seed=42)

        results[points] = {
            "objective": -float(
                model.log_likelihood()
            ),  # negative log marginal likelihood
            "rbf_variance": float(model.rbf.variance),
            "rbf_lengthscale": float(model.rbf.lengthscale),
            "noise_variance": float(model.Gaussian_noise.variance),
        }

    points = list(results.keys())
    fig, axs = plt.subplots(2, 2)

    axs[0, 0].plot(
        points, [results[M]["objective"] for M in points], marker="o", linewidth=2
    )
    axs[0, 0].set(xlabel="Nr. inducing points", ylabel="NLML")

    axs[0, 1].plot(
        points, [results[M]["rbf_variance"] for M in points], marker="o", linewidth=2
    )
    axs[0, 1].set(xlabel="Nr. inducing points", ylabel="RBF variance")

    axs[1, 0].plot(
        points, [results[M]["rbf_lengthscale"] for M in points], marker="o", linewidth=2
    )
    axs[1, 0].set(xlabel="Nr. inducing points", ylabel="RBF lengthscale")

    axs[1, 1].plot(
        points, [results[M]["noise_variance"] for M in points], marker="o", linewidth=2
    )
    axs[1, 1].set(xlabel="Nr. inducing points", ylabel="Noise variance")

    plt.tight_layout()
    plt.show()

    return results

def gaussian_process(
    x_data: np.ndarray,
    y_data: np.ndarray,
    kernel_variance: float = 1.0,
    kernel_width: float = 1.0,
    num_inducing: int = 50,
    max_iters: int = 100,
    seed: int = 42,
):
    # Ensures data input is correct
    if len(x_data.shape) == 1:
        x_data = x_data[:, None]
    if len(y_data.shape) == 1:
        y_data = y_data[:, None]

    # Check the validity of the number of inducing points
    num_inducing = num_inducing if num_inducing < x_data.shape[0] else x_data.shape[0]

    # Set seed for reproducibility
    np.random.seed(seed)

    # Define the RBF kernel
    input_dim = x_data.shape[1]
    kernel = GPy.kern.RBF(
        input_dim=input_dim, variance=kernel_variance, lengthscale=kernel_width
    )

    # Select M uniform spaced inducing points from the dataset
    inducing_indices = np.linspace(0, x_data.shape[0] - 1, num=num_inducing, dtype=int)
    selected_points = x_data[inducing_indices, :].copy()

    # Initialize a Sparse GP
    model = GPy.models.SparseGPRegression(
        x_data, y_data, kernel=kernel, Z=selected_points
    )

    # Optimize both kernel hyperparameters and inducing point locations
    model.optimize(
        "bfgs", max_iters=max_iters
    )  # minimizes the total negative log-likelihood

    # Plot the results
    if input_dim == 1:
        model.plot()
        plt.show()

    # Print the model parameters summary
    print(model)

    return model


if __name__ == "__main__":
    data, keys = load_npz_file()
    # print(f"Keys: {keys}")

    th_data = data["th"]  # Theta a.k.a. angular position (35000,)
    u_data = data["u"]  # Control input (35000,)

    # X_narx = (34998, 4), Y_narx = (34998,)
    X_narx, Y_narx = make_training_data(u_data, th_data, na=4, nb=4)

    # Split the data into a train, validation and test set
    x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = (
        train_test_val_split(X_narx, Y_narx)
    ) # x_train_raw = (0.6 * 34998, 4), y_train_raw = (0.6 * 34998,)

    # Normalize x values
    x_mean = np.mean(x_train_raw, axis=0)
    x_std = np.std(x_train_raw, axis=0)
    x_std[x_std == 0] = 1.0  # prevent zero division error

    x_train_scaled = (x_train_raw - x_mean) / x_std  # x_train_scaled = (20998, 4)
    x_val_scaled = (x_val_raw - x_mean) / x_std
    x_test_scaled = (x_test_raw - x_mean) / x_std

    # Normalize y values
    y_mean = np.mean(y_train_raw, axis=0)
    y_std = np.std(y_train_raw, axis=0)
    if y_std == 0:
        y_std = 1.0

    y_train_scaled = (y_train_raw - y_mean) / y_std  # y_train_scaled = (20998,)
    y_val_scaled = (y_val_raw - y_mean) / y_std
    y_test_scaled = (y_test_raw - y_mean) / y_std

    # Analyse the best amount of inducing points
    # results = inducing_point_search(x_train_scaled, y_train_scaled)

    # Train the GP model
    sparse_model = gaussian_process(x_train_scaled, y_train_scaled, num_inducing=50)

    # ----------- Predict using trained model ---------------------------------------
    y_train_pred_scaled, _ = sparse_model.predict(x_train_scaled)  # (20998, 1)
    y_train_pred = (y_train_pred_scaled * y_std) + y_mean
    y_train_pred = y_train_pred.flatten()  # (20998,)

    y_val_pred_scaled, _ = sparse_model.predict(x_val_scaled)
    y_val_pred = (y_val_pred_scaled * y_std) + y_mean
    y_val_pred = y_val_pred.flatten()  # (20998,)

    y_test_pred_scaled, _ = sparse_model.predict(x_test_scaled)
    y_test_pred = (y_test_pred_scaled * y_std) + y_mean
    y_test_pred = y_test_pred.flatten()  # (20998,)

    # ------------ Analyse performance ----------------------------------------------
    train_rms_rad = np.sqrt(np.mean((y_train_pred - y_train_raw) ** 2))
    val_rms_rad = np.sqrt(np.mean((y_val_pred - y_val_raw) ** 2))
    test_rms_rad = np.sqrt(np.mean((y_test_pred - y_test_raw) ** 2))

    summary_data = {
        "Dataset Split": ["Training Set", "Validation Set", "Test Set"],
        "RMS [rad]": [train_rms_rad, val_rms_rad, test_rms_rad],
        "RMS [deg]": [
            train_rms_rad / (2 * np.pi) * 360,
            val_rms_rad / (2 * np.pi) * 360,
            test_rms_rad / (2 * np.pi) * 360
        ],
        "NRMS [%]": [
            train_rms_rad / np.std(y_train_raw) * 100,
            val_rms_rad / y_val_raw.std() * 100,
            test_rms_rad / y_test_raw.std() * 100
        ]
    }

    metrics_df = pd.DataFrame(summary_data)
    print(metrics_df.to_string(index=False))

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
    plt.show()
   