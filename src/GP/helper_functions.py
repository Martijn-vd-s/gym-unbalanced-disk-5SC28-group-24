import os
import GPy
import pickle
import numpy as np
from matplotlib import pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR) # /GP
PARENT_DIR = os.path.dirname(SRC_DIR) # /src


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
    src_dir = os.path.dirname(script_dir)
    parent_dir = os.path.dirname(src_dir)

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
        Specifies the number of past input lags to include in the state feature vector.
    nb : int
        Specifies the number of past output lags to include in the state feature vector.

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
        x_lags = x_data[k - na : k]
        y_lags = y_data[k - nb : k]

        # Fuses x_slice and y_slice into a single 1D array
        Xdata.append(np.concatenate([x_lags, y_lags]))
        Ydata.append(y_data[k])
    return np.array(Xdata), np.array(Ydata)


def wrap_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

def save_model(model, path: str = None):
    """
    Saves model to pickle file

    Parameters
    ----------
    model:
        The model to be saved.
    path: str
        The path where the model is to be saved.
    """
    data_file = os.path.join(SRC_DIR, "sparse_gp_model.pkl")

    path = data_file if path is None else path

    # Save the model object
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_model(path: str = None):
    """
    Loads a model from a pickle file.

    Parameters
    ----------
    path: str
        The file path of the model.

    Returns:
        The loaded model
    """
    data_file = os.path.join(SCRIPT_DIR, "sparse_gp_model.pkl")
    path = data_file if path is None else path

    try:
        with open(path, "rb") as f:
            model = pickle.load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"The model file was not found at: {path}") from e
    except PermissionError as e:
        raise PermissionError(
            f"Unsufficient permission to open the file at: {path}"
        ) from e

    return model


def normalize_data(train, val, test):
    """
    Helper function to normalize the data.
    """
    mean = np.mean(train, axis=0)
    std = np.std(train, axis=0)

    std = np.where(std == 0, 1.0, std)  # prevents zero division

    train_scaled = (train - mean) / std
    val_scaled = (val - mean) / std
    test_scaled = (test - mean) / std

    return train_scaled, val_scaled, test_scaled, mean, std


def gaussian_process(
    x_data: np.ndarray,
    y_data: np.ndarray,
    kernel_variance: float = 1.0,
    kernel_width: float = 1.0,
    num_inducing: int = 50,
    max_iters: int = 100,
    seed: int = 42,
):
    """
    Trains a Sparse Gaussian Process Regression model using an RBF kernel.

    This function configures a Radial Basis Function (RBF) kernel with Automatic
    Relevance Determination (ARD) enabled, meaning a separate lengthscale is
    optimized for every input dimension (lag feature). It selects uniformly
    spaced inducing points across the input space to initialize the sparse
    approximations, and minimizes the negative log marginal likelihood using
    the BFGS optimization algorithm.

    Parameters
    ----------
    x_data : np.ndarray
        The input feature matrix of shape (Nsamples, Nfeatures) representing
        the NARX lagged state history.
    y_data : np.ndarray
        The target system state vector of shape (Nsamples, 1) representing
        the true system measurements (e.g., angular position).
    kernel_variance : float, default 1.0
        The vertical scale parameter (signal variance) of the RBF kernel,
        controlling the overall output variance amplitude.
    kernel_width : float, default 1.0
        The horizontal scale parameter (lengthscale) of the RBF kernel,
        determining how far data points can generalize across time/space features.
    num_inducing : int, default 50
        The number of pseudo-inputs (inducing points) used to compress the
        GP covariance matrix structure for computational efficiency. Capped at Nsamples.
    max_iters : int, default 100
        The maximum number of BFGS optimization steps allowed to fit the
        kernel hyperparameters and optimize inducing point coordinates.
    seed : int, default 42
        The random seed used to enforce exact reproducibility during initialization.

    Returns
    -------
    model : GPy.models.SparseGPRegression
        The optimized, trained Sparse Gaussian Process model instance.
    """
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
    # ARD: https://www.emergentmind.com/topics/automatic-relevance-determination-ard
    input_dim = x_data.shape[1]
    kernel = GPy.kern.RBF(
        input_dim=input_dim,
        variance=kernel_variance,
        lengthscale=kernel_width,
        ARD=True,  # A hierarchical Bayesian machine learning method used for automated feature selection and dimensionality reduction
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

    return model


def denormalize_data(train, val, test, mean, std):
    """
    Denormalizes data using mean and std.
    """
    train_unscaled = (train * std) + mean
    val_unscaled = (val * std) + mean
    test_unscaled = (test * std) + mean

    return train_unscaled, val_unscaled, test_unscaled
