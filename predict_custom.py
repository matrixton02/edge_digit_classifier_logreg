import cv2
import numpy as np
from model_utils import predict_multiclass

def preprocess_custom_digit(image_path):
    img=cv2.imread(image_path,cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    img_resized=cv2.resize(img,(8,8),interpolation=cv2.INTER_AREA)

    if(np.mean(img_resized)>127):
        img_resized=255-img_resized
    
    img_rescaled=(img_resized/255.0)*16.0
    edge_img=cv2.Canny(np.uint8(img_rescaled*(255.0/img_rescaled.max())),30,100)
    binary_map=(edge_img>0).astype(int)

    return binary_map.flatten(),img_resized

all_theta=np.load("saved/theta_all.npy")

X_custom, original_img = preprocess_custom_digit("data/test_image_1.jpg")
pred = predict_multiclass(np.array([X_custom]), all_theta)[0]

print("Predicted Digit:", pred)

# Show original image
import matplotlib.pyplot as plt
plt.imshow(original_img, cmap='gray')
plt.title(f"Predicted: {pred}")
plt.axis('off')
plt.show()