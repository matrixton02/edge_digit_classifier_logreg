import matplotlib.pyplot as plt
import numpy as np
import model_utils as mu
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def show_prediction(images,X_edge,y_true,y_pred,n=20):
    plt.figure(figsize=(12,6))
    count=0
    for i in range(len(y_true)):
        if(count==n):
            break
        plt.subplot(2,n//2,count+1)
        plt.imshow(images[i],cmap="gray")
        plt.axis("off")
        color="green" if y_pred[i]==y_true[i] else "red"
        plt.title(f"Pred: {y_pred[i]}\nTrue: {y_true[i]}",color=color)
        count+=1
    plt.suptitle("Model Predictions (Green = Correct, Red = Wrong)", fontsize=16)
    plt.tight_layout()
    plt.show()

all_theta=np.load("saved/theta_all.npy")

digits = load_digits()
images = digits.images
labels = digits.target

X=np.array([mu.extract_edges(img).flatten() for img in images])
y=labels.copy()

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)

y_pred_train = mu.predict_multiclass(X_train, all_theta)
y_pred_test = mu.predict_multiclass(X_test, all_theta)

train_acc = accuracy_score(y_train, y_pred_train)
test_acc = accuracy_score(y_test, y_pred_test)

print("Multiclass Train Accuracy:", train_acc)
print("Multiclass Test Accuracy:", test_acc)

sample_count=20
test_indices=np.random.choice(len(y_test),sample_count,replace=False)

test_imgs=[images[i] for i in test_indices]
X_test_sample=X_test[test_indices]
y_test_sample=y_test[test_indices]
y_pred_samples=mu.predict_multiclass(X_test_sample,all_theta)

show_prediction(test_imgs,X_test_sample,y_test_sample,y_pred_samples,n=sample_count)

