# 🧠 Edge-Based Digit Classifier using Logistic Regression

A compact handwritten digit recognizer that combines classical image processing with a simple machine learning model. This project uses **Canny edge detection** to extract stroke information from digit images, then trains a one-vs-all **logistic regression** classifier on these edge features.

> 💡 No deep learning required — just features + math.

---

## 🎯 Motivation

- **Highlight Shape Information:** Edge detection emphasizes digit strokes and boundaries, reducing irrelevant pixel information like brightness or background noise.
- **Simplicity and Interpretability:** Logistic regression is a linear, fast, and easy-to-understand classifier. It's perfect for understanding how different parts of an image influence classification.
- **Lightweight Solution:** Great for low-resource environments, fast experimentation, and educational projects.

---

## 🧠 Theory

### 🖼️ Canny Edge Detection
Canny is a multi-step algorithm that:
1. Smooths the image (Gaussian filter)
2. Computes gradients (Sobel filters)
3. Applies non-maximum suppression
4. Uses double thresholding
5. Tracks and connects edges

This transforms digit images into binary edge maps highlighting boundaries.

### 📈 One-vs-All Logistic Regression
We train 10 binary classifiers (one for each digit 0–9). Each model predicts whether a given input is the target digit or not. At inference, we use all 10 models and select the one with the highest probability.

---

## ⚙️ Setup Instructions

### 🔧 Requirements

- Python 3.7+
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib

Install all dependencies using:
```bash
pip install numpy opencv-python scikit-learn matplotlib
```
## 📦 Clone the Repository
```bash
git clone https://github.com/matrixton02/edge_digit_classifier_logreg.git
cd edge_digit_classifier_logreg
```

## 🏋️ Training the Model
Train the logistic regression classifier using the digits dataset:
```bash
python train_model.py
```
### This will:

- Load the 8×8 digit dataset from sklearn.datasets

- Convert each image into an edge map

- Train a logistic regression classifier per digit

- Save the model parameters to saved/theta_all.npy

## 🤖 Run Inference on a Custom Digit
To classify a custom digit image (e.g. a handwritten "3"):
```bash
python predict_custom.py data/test_image1.jpg
```

### Make sure the image is:

- Centered and grayscale

- Clean background (white or black)

- Clearly drawn single digit

## 📈 Model Performance

| Dataset       | Accuracy   |
|---------------|------------|
| Training Set  | 83.4%     |
| Test Set      | 80.55%     |

✅ These results were achieved using only logistic regression and edge maps — no deep learning involved.

---

## 📚 Learning Outcomes

- Implemented logistic regression from scratch
- Learned how edge maps capture digit structure
- Explored one-vs-all classification strategy
- Visualized preprocessing and prediction steps

---

## 📌 Credits

- **Dataset:** [Scikit-learn Digits Dataset](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html)
- **Edge Detection:** OpenCV’s Canny Filter
- **Inspiration:** Andrew Ng’s ML course, Scikit-learn examples

---

## 📬 Contact

**Yashasvi Kumar Tiwari**  
📧 yashasvikumartiwari@gmail.com  
🔗 [LinkedIn](https://www.linkedin.com/in/yashasvikumartiwari) | [GitHub](https://github.com/matrixton02)

