# Gaussian Process

Gaussian Processes (GP) are a nonparametric supervised learning method used to estimate functions directly from data. A dataset of an unbalanced disk was provided and utilized for system identification. The data was divided into a 60:20:20 split for training, validation, and testing respectively. Although Gaussian Processes offer accurate function approximation, they struggle with high dimensional data and have a computational bottleneck which makes them unsuitable for large dataset. Therefore a Sparse Gaussian Process (SGP) was used for the provided data.

## File Tree
```
GP/
├── __pycache__/
├── csv_files/                      # CSV files containing training results
├── images/                         
├── __init__.py
├── helper_functions.py             # Contains helper functions
├── hyperparameter_search.py        # Runs a grid search for hyperparameters
├── README.md
├── scaler_params.npz               # Contains data normalization parameters
├── simulate_gp.py                  # Runs one-step-ahead and free run simulation prediction
├── sparse_gp_model.pkl             # Contains the last trained sparse gaussian process
└── train_gp.py                     # Trains the sparse gaussian process
```