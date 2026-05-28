# pylint: disable=invalid-name

import time
import numpy as np
import gym_unbalanced_disk
from collections import deque
import helper_functions as hf
import matplotlib.pyplot as plt

if __name__ == "__main__":
    model = hf.load_model()
    print(model)

    try:
        scaler = np.load("scaler_params.npz")
        x_mean = scaler["x_mean"]
        x_std = scaler["x_std"]
        y_mean = scaler["y_mean"]
        y_std = scaler["y_std"]
    except FileNotFoundError:
        print(
            "ERROR: 'scaler_params.npz' not found! Make sure you ran your training script to save them."
        )
        exit()

    # Define input and output lag
    na = 2  # Input lags
    nb = 8  # Output lags

    # input & output buffer
    u_history = deque([0.0] * na, maxlen=na)
    theta_history = deque([0.0] * nb, maxlen=nb)
    model_theta_history = deque([0.0] * nb, maxlen=nb)

    env = gym_unbalanced_disk.UnbalancedDisk(dt=0.025, umax=3.0)
    obs, info = env.reset()
    th_noise, omega_noise = obs

    # Lists to store history for plotting
    steps = []
    true_thetas = []
    sim_thetas = []
    std_deviations = []

    # Warmup used to fill the input & output buffer
    warmup_steps = max(na, nb)

    for step in range(warmup_steps):
        # Generate an action
        action = env.action_space.sample()
        u_input = float(action)

        # Step the environment to collect output
        obs, reward, terminated, truncated, info = env.step(float(action))
        th_noise, omega_noise = obs

        # Add the values to the buffers
        u_history.append(u_input)
        theta_history.append(th_noise)
        model_theta_history.append(th_noise)

    try:
        for i in range(200):
            action = env.action_space.sample()
            u_input = float(action)

            osa = True
            if osa:
                X_raw = np.concatenate([list(u_history), list(theta_history)]).reshape(
                    1, -1
                )  # One-step ahead simulation
                name = "images/gp_simulation_performance_osa.png"
                label = "GP One-Step-Ahead Prediction (Mean)"
                title = "One-Step-Ahead Gaussian Process Simulation Evaluation"
            else:
                X_raw = np.concatenate(
                    [list(u_history), list(model_theta_history)]
                ).reshape(1, -1)  # Free run simulation
                name = "images/gp_simulation_performance_fr.png"
                label = "GP Free-Run Prediction (Mean)"
                title = "Free-Run Gaussian Process Simulation Evaluation"

            X_scaled = (X_raw - x_mean) / x_std

            gp_mean, gp_var = model.predict(X_scaled)

            predicted_theta_raw = (float(np.squeeze(gp_mean)) * y_std) + y_mean
            predicted_theta = hf.wrap_angle(predicted_theta_raw)
            variance_theta = float(np.squeeze(gp_var))
            std_raw = np.sqrt(variance_theta) * y_std

            # Advance the environment
            obs, reward, terminated, truncated, info = env.step(u_input)
            th_noise, omega_noise = obs  # Extract true position

            error = abs(th_noise - predicted_theta)
            print(
                f"Step {i:3d} | True θ: {th_noise:7.4f} | GP Simulated θ: {predicted_theta:7.4f} | Cum. Error: {error:7.5f}"
            )

            # Store data for plotting
            steps.append(i)
            true_thetas.append(th_noise)
            sim_thetas.append(predicted_theta)
            std_deviations.append(std_raw)

            # Update buffers
            u_history.append(u_input)
            theta_history.append(th_noise)
            model_theta_history.append(predicted_theta)

            env.render()
            time.sleep(1 / 24)
            if terminated or truncated:
                obs = env.reset()
    finally:  # this will always run
        env.close()

    # --- PLOTTING CODE ---
    steps = np.array(steps)
    true_thetas = np.array(true_thetas)
    sim_thetas = np.array(sim_thetas)
    std_deviations = np.array(std_deviations)

    # Calculate 95% confidence intervals (+/- 2 standard deviations)
    lower_bound = sim_thetas - 2 * std_deviations
    upper_bound = sim_thetas + 2 * std_deviations

    plt.figure(figsize=(12, 6))

    # Plot true trajectory
    plt.plot(
        steps,
        true_thetas,
        label="Simulation Values",
        color="black",
        linestyle="--",
        linewidth=1.5,
    )

    # Plot GP mean prediction
    plt.plot(
        steps,
        sim_thetas,
        label=label,
        color="blue",
        linewidth=2,
    )

    # Shade the 95% confidence interval
    plt.fill_between(
        steps,
        lower_bound,
        upper_bound,
        color="blue",
        alpha=0.15,
        label="95% Confidence Interval (±2σ)",
    )

    plt.title(title, fontsize=14)
    plt.xlabel("Simulation Time Step", fontsize=12)
    plt.ylabel("Disk Angle (radians)", fontsize=12)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", fontsize=11)

    # Optional: Plot the absolute error on a secondary axis or separate subplot
    plt.tight_layout()

    plt.savefig(name, dpi=300)
    plt.show()
