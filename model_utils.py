import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def gradient(X, theta, y):
    m = y.size
    return (X.T @ (sigmoid(X @ theta) - y)) / m

def gradient_descent(X, y, alpha=0.1, num_iter=1000, tol=1e-7):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    theta = np.zeros(X_b.shape[1])
    for _ in range(num_iter):
        grad = gradient(X_b, theta, y)
        theta -= alpha * grad
        if np.linalg.norm(grad) < tol:
            break
    return theta

def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict_multiclass(X, all_theta):
    probs = np.array([predict_prob(X, theta) for theta in all_theta])
    return np.argmax(probs, axis=0)