# pylint: disable=import-error
# pylint: disable=invalid-name

import pandas as pd
import numpy as np
import helper_functions as hf


def fill_prediction_table():
    """
    Fills the target column of `hidden-test-prediction-submission-file.csv` using the trained Sparse GP model.
    """
    # Load file
    csv_path = "hidden-test-prediction-submission-file.csv"
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    # Load trained Sparse GP model
    model = hf.load_model()

    try:
        scaler = np.load("scaler_params.npz")
        input_mean = scaler["x_mean"]
        input_std = scaler["x_std"]
        output_mean = scaler["y_mean"]
        output_std = scaler["y_std"]
    except FileNotFoundError:
        print("ERROR: 'scaler_params.npz' not found!")
        exit()

    # Define input and output lag
    na = 2  # Input lags
    nb = 8  # Output lags

    # Restructure input and output
    # ['u[k-2]', 'u[k-1]'] - columns for input lags
    u_cols = [f"u[k-{i}]" for i in range(na, 0, -1)]
    # ['y[k-8]', ..., 'y[k-1]'] - columns for output lags
    y_cols = [f"y[k-{i}]" for i in range(nb, 0, -1)]

    # Extract data from the columns
    X_u = df[u_cols].values
    X_y = df[y_cols].values

    # Create the input matrix for prediction
    X_input = np.hstack([X_u, X_y])

    # Normalization
    X_submission = (X_input - input_mean) / input_std

    # Generate predictions using the GP model
    y_pred_mean, y_pred_var = model.predict(X_submission)

    # Normalize the outputted predictions
    y_pred_mean = (y_pred_mean * output_std) + output_mean

    # Update the table's target column
    df["y[k-0]"] = y_pred_mean.flatten()

    df.to_csv(csv_path, index=False)
    print(f"Successfully generated predictions and saved to {csv_path}")


def fill_simulation_table():
    """
    Fills the target column of `hidden-test-simulation-submission-file.csv` using the trained Sparse GP model.
    """
    # Load file
    csv_path = "hidden-test-simulation-submission-file.csv"
    df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    # Load trained Sparse GP model
    model = hf.load_model()

    try:
        scaler = np.load("scaler_params.npz")
        input_mean = scaler["x_mean"]
        input_std = scaler["x_std"]
        output_mean = scaler["y_mean"]
        output_std = scaler["y_std"]
    except FileNotFoundError:
        print("ERROR: 'scaler_params.npz' not found!")
        exit()

    # Define input and output lag
    na = 2  # Input lags
    nb = 8  # Output lags

    u_values, th_values = df[df.columns.to_list()].values.T
    total_steps = len(df)

    print(total_steps)

    num_warm_up_steps = 50

    for i in range(num_warm_up_steps, total_steps):
        # Prepare input and output lags
        u_lags = u_values[i - na : i]
        y_lags = th_values[i - nb : i]

        # Create model input
        X_input = np.hstack([u_lags, y_lags]).reshape(1, -1)

        # Normalize input
        X_scaled = (X_input - input_mean) / input_std

        # Predict next output
        y_pred, _ = model.predict(X_scaled)

        # Denormalize model output
        y_pred = (y_pred * output_std) + output_mean

        # Update the target column with the predicted value
        th_values[i] = y_pred.item()
    
    df["th"] = th_values
    df.to_csv(csv_path, index=False)


if __name__ == "__main__":
    # fill_prediction_table()
    fill_simulation_table()
