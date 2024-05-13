import matplotlib.pyplot as plt
import numpy as np

def get_mse(X: np.array, theta: np.array, y_true: np.array):
    """
    Get the mean squared error between predicted and actual values.
    """
    y_hat = np.dot(X, theta)
    return np.mean(np.power(y_true - y_hat, 2))


def plot_loss_curve(cost_history:list[float], show_plot:bool=False):
    """
    Plot the loss curve.
    """
    plt.plot(cost_history)
    plt.xlabel('Epoch')
    plt.ylabel('Cost')
    plt.title('Loss Curve')
    if show_plot:
        plt.show()
    plt.savefig('../plots/loss_curve.png')
    plt.close()

def plot_average_error(X:np.array, y_test:np.array, theta:np.array, show_plot:bool=False):
    """
    Plot the average error over the test set.
    """
    y_hat = np.dot(X, theta)
    y_diff = abs(y_hat-y_test)
    plt.plot(y_diff)
    plt.hlines(y=np.average(y_diff), xmin=0, xmax=len(y_diff), color='r', label=f'Average Error: {np.average(y_diff):2f}')
    plt.xlabel('Sample')
    plt.ylabel('Absolute Error')
    plt.title('Average Error')
    plt.legend()
    if show_plot:
        plt.show()
    plt.savefig('../plots/average_error.png')
    plt.close()
