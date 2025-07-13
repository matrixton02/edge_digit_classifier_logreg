import cv2
import numpy as np
from model_utils import predict_multiclass
import matplotlib.pyplot as plt

def preprocess_custom_digit(image_path):
    img_original=cv2.imread(image_path)

    if img_original is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    
    img_gray=cv2.cvtColor(img_original,cv2.COLOR_BGR2GRAY)    
    img_resized=cv2.resize(img_gray,(8,8),interpolation=cv2.INTER_AREA)

    if(np.mean(img_resized)>127):
        img_resized=255-img_resized
    
    img_rescaled=(img_resized/255.0)*16.0
    edge_img=cv2.Canny(np.uint8(img_rescaled*(255.0/img_rescaled.max())),30,100)
    binary_map=(edge_img>0).astype(int)

    return binary_map.flatten(),img_original,img_resized,edge_img

all_theta=np.load("saved/theta_all.npy")

image_path="data/test_image_1.jpg"
X_custom, img_original,img_resized,edge_img = preprocess_custom_digit(image_path)
pred = predict_multiclass(np.array([X_custom]), all_theta)[0]

plt.figure(figsize=(12,4))

plt.subplot(1,3,1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(img_original,cv2.COLOR_BGR2RGB))
plt.axis("off")

plt.subplot(1,3,2)
plt.title("Grayscalev(8x8)")
plt.imshow(img_resized,cmap="gray")
plt.axis("off")

plt.subplot(1,3,3)
plt.title("Edge Map")
plt.imshow(edge_img,cmap="gray")
plt.axis("off")

plt.suptitle(f"Predicted: {pred}")

plt.tight_layout()
plt.show()

print("Predicted Digit:", pred)