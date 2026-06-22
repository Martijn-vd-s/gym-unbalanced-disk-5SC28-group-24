# pylint: disable=invalid-name
# pylint: disable=redefined-outer-name

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

import helper_functions as hf


def inducing_point_search(X_narx, Y_narx):
    """
    Grid search to find the optimal number of inducing points for the sparse Gaussian Process.
    Handles data splitting and normalization internally.

    Parameters
    ----------
    X_narx:
        The input data from the NARX.
    Y_narx:
        The output data from the NARX.

    Returns
    -------
        The statistical results from the grid search.
    """
    # Split the NARX data
    x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = (
        hf.train_test_val_split(X_narx, Y_narx)
    )

    # Normalize the data
    x_train_scaled, x_val_scaled, x_test_scaled, _, _ = hf.normalize_data(
        x_train_raw, x_val_raw, x_test_raw
    )
    y_train_scaled, _, _, y_mean, y_std = hf.normalize_data(
        y_train_raw, y_val_raw, y_test_raw
    )

    nr_points = [10, 20, 30, 40, 50, 75, 100, 125, 150]
    best_val_error = float("inf")
    # best_ind_points = nr_points[0]
    results = {}

    for points in nr_points:
        print(points)
        model = hf.gaussian_process(
            x_train_scaled, y_train_scaled, num_inducing=points, seed=42
        )

        y_train_pred_scaled, _ = model.predict(x_train_scaled)
        y_val_pred_scaled, _ = model.predict(x_val_scaled)
        y_test_pred_scaled, _ = model.predict(x_test_scaled)

        y_train_pred, y_val_pred, y_test_pred = hf.denormalize_data(
            y_train_pred_scaled,
            y_val_pred_scaled,
            y_test_pred_scaled,
            y_mean,
            y_std,
        )

        y_train_pred = y_train_pred.flatten()
        y_val_pred = y_val_pred.flatten()
        y_test_pred = y_test_pred.flatten()

        # Calculate RMS errors
        train_rms_rad = np.sqrt(np.mean((y_train_pred - y_train_raw) ** 2))
        val_rms_rad = np.sqrt(np.mean((y_val_pred - y_val_raw) ** 2))
        test_rms_rad = np.sqrt(np.mean((y_test_pred - y_test_raw) ** 2))

        if val_rms_rad < best_val_error:
            best_val_error = val_rms_rad
            # best_ind_points = points

        results[points] = {
            "objective": -float(model.log_likelihood()),
            "noise_variance": float(model.Gaussian_noise.variance.values[0]),
            "train_rms": train_rms_rad,
            "val_rms": val_rms_rad,
            "test_rms": test_rms_rad,
        }

    df_results = pd.DataFrame.from_dict(results, orient="index")

    return df_results


def search_optimal_lag():
    """
    Grid search to find the optimal lag parameters.
    """
    lag_options = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    best_val_error = float("inf")
    best_na, best_nb = 1, 1

    data, _ = hf.load_npz_file()
    th_data = data["th"]  # Theta a.k.a. angular position (35000,)
    u_data = data["u"]  # Control input (35000,)

    results = {}

    for na in lag_options:
        for nb in lag_options:
            X_narx, Y_narx = hf.make_training_data(u_data, th_data, na=na, nb=nb)

            x_train_raw, y_train_raw, x_val_raw, y_val_raw, x_test_raw, y_test_raw = (
                hf.train_test_val_split(X_narx, Y_narx)
            )

            x_train_scaled, x_val_scaled, x_test_scaled, _, _ = (
                hf.normalize_data(x_train_raw, x_val_raw, x_test_raw)
            )

            y_train_scaled, _, _, y_mean, y_std = (
                hf.normalize_data(y_train_raw, y_val_raw, y_test_raw)
            )

            model = hf.gaussian_process(
                x_train_scaled, y_train_scaled, num_inducing=50, max_iters=30
            )

            y_train_pred_scaled, _ = model.predict(x_train_scaled)  # (20998, 1)
            y_val_pred_scaled, _ = model.predict(x_val_scaled)
            y_test_pred_scaled, _ = model.predict(x_test_scaled)

            y_train_pred, y_val_pred, y_test_pred = hf.denormalize_data(
                y_train_pred_scaled,
                y_val_pred_scaled,
                y_test_pred_scaled,
                y_mean,
                y_std,
            )

            y_train_pred = y_train_pred.flatten()  # (20998,)
            y_val_pred = y_val_pred.flatten()  # (20998,)
            y_test_pred = y_test_pred.flatten()  # (20998,)

            train_rms_rad = np.sqrt(np.mean((y_train_pred - y_train_raw) ** 2))
            val_rms_rad = np.sqrt(np.mean((y_val_pred - y_val_raw) ** 2))
            test_rms_rad = np.sqrt(np.mean((y_test_pred - y_test_raw) ** 2))

            print(f"Lags na={na}, nb={nb} --> Val RMS: {val_rms_rad:.4f}")

            if val_rms_rad < best_val_error:
                best_val_error = val_rms_rad
                best_na = na
                best_nb = nb

            results[(na, nb)] = {
                "objective": -float(model.log_likelihood()),
                "rbf_variance": float(model.rbf.variance),
                "rbf_lengthscale": model.rbf.lengthscale.tolist(),
                "noise_variance": float(model.Gaussian_noise.variance),
                "train_rms": float(train_rms_rad),
                "val_rms": float(val_rms_rad),
                "test_rms": float(test_rms_rad),
            }

    df = pd.DataFrame.from_dict(results, orient="index")
    df.index.names = ["na", "nb"]
    df = df.reset_index()

    print(f"Best na: {best_na} & nb: {best_nb}")
    return best_na, best_nb, df


if __name__ == "__main__":
    # ---------- Analyze inducing points results ---------------------------------
    data, keys = hf.load_npz_file()
    th_data = data["th"]  # Theta a.k.a. angular position (35000,)
    u_data = data["u"]  # Control input (35000,)

    X_narx, Y_narx = hf.make_training_data(
        u_data, th_data, na=4, nb=4
    )  # X_narx = (34998, 4), Y_narx = (34998,)

    inducing_results = inducing_point_search(X_narx, Y_narx)
    print(inducing_results.to_string(index=False))
    inducing_results.to_csv("csv_files/inducing_points_results.csv", index=True)

    # Create the figure subplots
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    axs[0, 0].plot(
        inducing_results.index, inducing_results["objective"], marker="o", linewidth=2
    )
    axs[0, 0].set(xlabel="Nr. inducing points", ylabel="NLML")

    axs[0, 1].plot(
        inducing_results.index,
        inducing_results["noise_variance"],
        marker="o",
        linewidth=2,
        color="purple",
    )
    axs[0, 1].set(xlabel="Nr. inducing points", ylabel="Noise variance")

    axs[1, 0].plot(
        inducing_results.index,
        inducing_results["train_rms"],
        marker="o",
        linewidth=2,
        color="orange",
    )
    axs[1, 0].set(xlabel="Nr. inducing points", ylabel="Train RMS (rad)")

    axs[1, 1].plot(
        inducing_results.index,
        inducing_results["val_rms"],
        marker="o",
        linewidth=2,
        color="green",
    )
    axs[1, 1].set(xlabel="Nr. inducing points", ylabel="Validation RMS (rad)")

    plt.tight_layout()
    plt.savefig("images/inducing_points_analysis.png", dpi=300, bbox_inches="tight")
    # plt.show()

    # ---------- Analyze NARX input and output lag ---------------------------------
    _, _, lag_results = search_optimal_lag()

    print(lag_results.to_string(index=False))
    lag_results.to_csv("csv_files/lag_results.csv", index=True)

    train_pivot = lag_results.pivot(index="na", columns="nb", values="train_rms")
    val_pivot = lag_results.pivot(index="na", columns="nb", values="val_rms")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Train Heatmap
    sns.heatmap(
        train_pivot,
        cmap="viridis_r",
        annot=True,
        fmt=".4f",
        linewidths=0.5,
        cbar_kws={"label": "RMS Error (rad)"},
        ax=ax1,
    )
    ax1.set_title("Train RMS across Lags", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Output Lag (nb)", fontsize=11)
    ax1.set_ylabel("Input Lag (na)", fontsize=11)
    ax1.invert_yaxis()  # Puts smaller lags (e.g. 2) at the bottom

    # Validation Heatmap
    sns.heatmap(
        val_pivot,
        cmap="viridis_r",
        annot=True,
        fmt=".4f",
        linewidths=0.5,
        cbar_kws={"label": "RMS Error (rad)"},
        ax=ax2,
    )
    ax2.set_title("Validation RMS across Lags", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Output Lag (nb)", fontsize=11)
    ax2.set_ylabel("Input Lag (na)", fontsize=11)
    ax2.invert_yaxis()

    plt.tight_layout()
    plt.savefig("images/lag_analysis.png", dpi=300, bbox_inches="tight")
    # plt.show()