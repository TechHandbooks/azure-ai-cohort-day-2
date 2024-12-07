### **Day 2 Assignment: Build a Machine Learning Model to Classify Handwritten Digits**

---

#### **Objective**
The goal of this assignment is to build a machine learning model to classify handwritten digits using the **digits dataset** from scikit-learn. This will help you apply the concepts learned in the session and get hands-on experience with data preprocessing, model training, and evaluation.

---

#### **Steps to Complete the Assignment**

1. **Understand the Digits Dataset**
   - The digits dataset contains 8x8 grayscale images of handwritten digits (0-9) and their corresponding labels.
   - Each image is represented as a flattened array of 64 features (pixels).
   - Learn more: [scikit-learn Digits Dataset Documentation](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html).

2. **Load and Explore the Dataset**
   - Use scikit-learn's `load_digits` function to load the dataset.
   - Visualize a few samples using **matplotlib**.

3. **Split Data**
   - Divide the dataset into training and testing sets using `train_test_split` from scikit-learn.

4. **Train a Model**
   - Use a classifier like **RandomForestClassifier** or any other model you prefer from scikit-learn.

5. **Evaluate the Model**
   - Check the model's accuracy using metrics like **accuracy_score**.
   - Generate and visualize a confusion matrix.

6. **Document Your Observations**
   - Discuss your findings, such as accuracy, challenges faced, and improvements tried.

7. **Push Your Work to GitHub**
   - Create a new repository named `Digits-Classification`.
   - Upload the following:
     - Python script or Jupyter notebook.
     - A README.md file with a brief description of the project and how to run it.
     - Observations and results (text or screenshots).

---

#### **Template Code**

Here’s a starting point for your assignment:

```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Step 1: Load the dataset
digits = load_digits()
X = digits.data  # Features
y = digits.target  # Labels

# Step 2: Visualize some samples
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(digits.images[i], cmap='gray')
    plt.title(f"Label: {digits.target[i]}")
    plt.axis('off')
plt.show()

# Step 3: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 4: Train a model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions and evaluate
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 6: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=digits.target_names)
disp.plot(cmap='viridis')
plt.show()
```

---

#### **Submission Instructions**

1. **Create a GitHub Repository**
   - Name the repository: `Digits-Classification`.
   - Add a meaningful description, e.g., "A machine learning project to classify handwritten digits using scikit-learn."

2. **Upload Files**
   - Python script or Jupyter Notebook.
   - README.md with the following details:
     - Project Overview
     - Steps to Run the Code
     - Results and Observations
   - Screenshots of model performance and visualizations (optional).

3. **Share the Repository Link**
   - Once your work is uploaded, share the repository URL with your instructor.

---

#### **Evaluation Criteria**
- Completeness: Have you followed all steps mentioned?
- Code Quality: Is the code readable, commented, and functional?
- Observations: Have you clearly explained your findings and results?
- GitHub Usage: Is the repository well-structured and documented?

---

#### **Bonus**
- Experiment with different models (e.g., Support Vector Machines, Logistic Regression).
- Tune hyperparameters to improve accuracy.

Good luck with your assignment! 🚀