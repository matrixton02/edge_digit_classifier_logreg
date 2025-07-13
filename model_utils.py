import numpy as np
import cv2
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

def extract_edges(img):
    img = np.uint8(img * (255.0 / img.max()))  # Scale to 0-255
    edge_img = cv2.Canny(img, threshold1=30, threshold2=100)
    return (edge_img > 0).astype(int)

def predict_prob(X, theta):
    X_b = np.c_[np.ones((X.shape[0], 1)), X]
    return sigmoid(X_b @ theta)

def predict_multiclass(X, all_theta):
    probs = np.array([predict_prob(X, theta) for theta in all_theta])
    return np.argmax(probs, axis=0)