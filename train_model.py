import numpy as np
import cv2
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from model_utils import gradient_descent, predict_multiclass
from sklearn.metrics import accuracy_score
import os

def extract_edges(img):
    img = np.uint8(img * (255.0 / img.max()))
    edge_img = cv2.Canny(img, 30, 100)
    return (edge_img > 0).astype(int)

digits = load_digits()
images = digits.images
labels = digits.target

X = np.array([extract_edges(img).flatten() for img in images])
y = labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train One-vs-All Logistic Regression
num_classes = 10
all_theta = []
for c in range(num_classes):
    y_binary = (y_train == c).astype(int)
    theta_c = gradient_descent(X_train, y_binary, alpha=0.01, num_iter=3000)
    all_theta.append(theta_c)
all_theta = np.array(all_theta)

# Evaluate
y_pred_train = predict_multiclass(X_train, all_theta)
y_pred_test = predict_multiclass(X_test, all_theta)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("Multiclass Train Accuracy:", train_acc)
print("Multiclass Test Accuracy:", test_acc)

# Save model
os.makedirs("saved", exist_ok=True)
np.save("saved/theta_all.npy", all_theta)
print("Saved model to 'saved/theta_all.npy'")