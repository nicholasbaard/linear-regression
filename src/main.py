import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import preprocess
from linear_regression import gradient_descent
from utils import get_mse, plot_loss_curve, plot_average_error

import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='../data/Walmart.csv', help='Path to the dataset')
    parser.add_argument('--cols_to_drop', type=list, default=['Date', 'Store'], help='Columns to drop')
    parser.add_argument('--target_col', type=str, default='Weekly_Sales', help='Target column')
    parser.add_argument('--alpha', type=float, default=0.001, help='Learning rate for gradient descent')
    parser.add_argument('--eps', type=float, default=0.0001, help='Convergence threshold for gradient descent')
    parser.add_argument('--lambda', type=float, default=0.001, help='Regularization parameter')
    parser.add_argument('--scaler_type', type=str, default='minmax', help='Scaler type - minmax or standardization')
    parser.add_argument('--show_plot', type=bool, default=False, help='Whether you want the plots to be shown')
    args = vars(parser.parse_args())

    # Import dataset:
    df = pd.read_csv(args['dataset_path'])

    # preprocess the data
    X_train, X_test, y_train, y_test, theta = preprocess(df, 
                                                        cols_to_drop=args['cols_to_drop'], 
                                                        target_col=args['target_col'],
                                                        scaler_type=args['scaler_type']
                                                        )


    # Train the model
    theta, cost_history = gradient_descent(X_train, y_train, theta, args['alpha'], args['eps'], args['lambda'])

    # Print the final weights and cost
    print(f"Final weights: {theta.ravel()}")
    print(f"Final cost: {cost_history[-1]}")

    # Print training MSE
    training_error = get_mse(X_train, theta, y_train)
    print(f"Training mean squared error: {training_error}")

    # Plot the cost history
    plot_loss_curve(cost_history, show_plot=args['show_plot'])

    # Print test MSE
    test_error = get_mse(X_test, theta, y_test)
    print(f"Test mean squared error: {test_error}")

    # Plot errors on test set
    plot_average_error(X_test, y_test, theta, show_plot=args['show_plot'])
    