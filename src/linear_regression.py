import numpy as np

# Compute the cost function
def compute_cost(X:np.array, y:np.array, theta:np.array):
    n = len(y)

    # Vectorized implementation of the cost function
    J = 1 / (2 * n) * np.sum(np.square(np.dot(X, theta) - y))
    
    return J

# Gradient descent function
def gradient_descent(X:np.array, y:np.array, theta:np.array, alpha:float=0.01, eps:float=0.0001, lmbda:float=0.001):
    m = len(y)
    cost_history = []
    theta_old = np.zeros_like(theta)

    #loop until convergence:
    while np.sqrt(np.sum(np.power(theta - theta_old, 2))) > eps:
        theta_old = theta

        # Compute the gradients
        gradients = 1/m * np.dot(X.T, np.dot(X, theta) - y) + (lmbda * np.sum(np.power(theta, 2)))
        
        # Update the weights
        theta = theta - alpha * gradients

        # Compute the cost
        cost_history.append(compute_cost(X, y, theta))
    
    return theta, cost_history
