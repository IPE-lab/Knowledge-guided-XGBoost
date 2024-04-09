# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 11:01:09 2024

@author: lenovo
"""

import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import matplotlib.pyplot as plt
from sklearn.model_selection import RandomizedSearchCV

# Function to load and preprocess data
def load_and_preprocess_data(file_path):
    data = pd.read_csv(file_path, index_col=0, header=0)
    X = data.iloc[:, 0:5].values
    y_i = data.iloc[:, 5].values
    y_emp = data.iloc[:, 6].values
    X_normalized, x_mean, x_std = norm(X)
    y_i_normalized, y_mean, y_std = norm(y_i)
    y_emp_normalized, _, _ = norm(y_emp)
    return X, y_i, y_emp, x_mean, x_std, y_mean, y_std

# Function to normalize data
def norm(x):
    x = np.array(x)
    x1 = (x - np.mean(x)) / np.std(x)
    return x1, np.mean(x), np.std(x)

# Function to split data into train and test sets
def split_data(X, y_i, y_emp, test_size=0.6, random_state=42):
    X_train, X_test, y_i_train, y_i_test, y_emp_train, y_emp_test = train_test_split(
        X, y_i, y_emp, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_i_train, y_i_test, y_emp_train, y_emp_test

# Function to calculate alpha_DK and gamma_DK
def calculate_alpha_gamma(y_i_train, y_emp_train):
    alpha_DK = np.abs((y_i_train - y_emp_train) / y_i_train)
    gamma_DK = len(alpha_DK) / np.sum(alpha_DK)
    return alpha_DK, gamma_DK

# Function to create and train the XGBoost model
def train_xgboost_model(X_train, y_i_train, alpha_DK, gamma_DK, num_rounds=100):
    dtrain = xgb.DMatrix(X_train, label=y_i_train)
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
    }
    def custom_loss(preds, dtrain):
        y_label1 = dtrain.get_label()
        weights = alpha_DK * gamma_DK
        grad = -2 * weights * (y_label1 - preds)
        hess = 2 * weights
        return grad, hess
    bst = xgb.train(params, dtrain, num_rounds, obj=custom_loss)
    return bst

# Function to evaluate the model
def evaluate_model(bst, X_test, y_i_test):
    dtest = xgb.DMatrix(X_test, label=y_i_test)
    y_test_pred = bst.predict(dtest)
    mse = mean_squared_error(y_i_test, y_test_pred)
    mape = mean_absolute_percentage_error(y_i_test, y_test_pred) * 100
    r2 = r2_score(y_i_test, y_test_pred)
    return mse, mape, r2, y_test_pred

# Function to plot the results
def plot_results(y_i_test, y_test_pred):
    min_val = min(np.min(y_i_test), np.min(y_test_pred))
    max_val = max(np.max(y_i_test), np.max(y_test_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle='-', color='black', linewidth=2, label='45-degree diagonal')
    plt.scatter(y_i_test, y_test_pred, alpha=0.5, color='blue', label='Test data')
    plt.title('Model Fit on Test Data')
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.show()

# Main function to run the entire process
def main(file_path):
    # Load and preprocess data
    X_normalized, y_i_normalized, y_emp_normalized, x_mean, x_std, y_mean, y_std = load_and_preprocess_data(file_path)
    

    # Split data
    X_train, X_test, y_i_train, y_i_test, y_emp_train, y_emp_test = split_data(X_normalized, y_i_normalized, y_emp_normalized,test_size=0.6,random_state=42)
    X_train, X_test, y_i_train, y_i_test, y_emp_train, y_emp_test = split_data(
        X_test, y_i_test, y_emp_test, test_size=0.35, random_state=42
    )
    
    # Calculate alpha_DK and gamma_DK
    alpha_DK, gamma_DK = calculate_alpha_gamma(y_i_train, y_emp_train)
    
    # Train the XGBoost model
    bst = train_xgboost_model(X_train, y_i_train, alpha_DK, gamma_DK)

    # Evaluate the model
    mse, mape, r2, y_test_pred = evaluate_model(bst, X_test, y_i_test)

    # Print evaluation metrics
    print(f'Mean Squared Error (MSE): {mse:.4f}')
    print(f'R-squared (R²): {r2:.4f}')
    print(f'Mean Absolute Percentage Error (MAPE): {mape:.4f}')

    # Plot the results
    plot_results(y_i_test, y_test_pred)

# Run the main function
if __name__ == '__main__':
    file_path = "Data.csv"
    main(file_path)
