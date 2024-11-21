# Left Hand, Right Hand, and Both Hands Classification from Images

## Introduction
This project utilizes image processing and machine learning techniques to classify images of left hands, right hands, or both hands. Data is processed from an image dataset and labeled information to train an SVM model for high accuracy.

## Requirements
- Python 3.x
- Python libraries:
  - OpenCV
  - NumPy
  - Pandas
  - Scikit-learn
  - Matplotlib
  - Joblib
  - PIL (Pillow)

## Workflow
1. **Data Preparation**:
   - Read and process a CSV file containing image labels.
   - Classify images into left hand, right hand, and both hands categories.
2. **Image Preprocessing**:
   - Convert images to grayscale.
   - Resize images to 64x64 pixels.
3. **Model Training**:
   - Use an SVM model with a linear kernel.
   - Split the data into training (80%) and testing (20%) sets.
4. **Model Evaluation**:
   - Evaluate model accuracy using metrics such as `accuracy`, `precision`, `recall`, and `f1-score`.
   - Display classification reports and misclassified images.
5. **Model Saving**:
   - Save the trained model to a `.pkl` file.

## How to Use
1. **Set up the environment**:
   - Install Python and required libraries:
     ```bash
     pip install opencv-python numpy pandas scikit-learn matplotlib joblib pillow
     ```
2. **Run the program**:
   - Ensure image data and the CSV file are in the correct directories.
   - Run the Python script to train the model:
     ```bash
     python hand_classification.py
     ```
3. **Test the model**:
   - Use the test set to evaluate accuracy and view predictions.

## Results
- **Accuracy**: The model achieves 96.80% accuracy.
- **Misclassified images**: 16 images were misclassified.
- **Classification report**:
  ```
              precision    recall  f1-score   support

           2       0.00      0.00      0.00         1
           L       0.00      0.00      0.00         1
           P       0.75      0.60      0.67        20
           T       0.98      0.99      0.98       478
           p       0.00      0.00      0.00         0

   micro avg       0.97      0.97      0.97       500
   macro avg       0.35      0.32      0.33       500
  weighted avg     0.97      0.97      0.97       500
  ```
