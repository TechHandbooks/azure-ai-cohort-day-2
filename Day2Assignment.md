### **Day 2 Assignment: Build a Machine Learning Model to Classify Handwritten Digits**

---

#### **Objective**
The goal of this assignment is to build a machine learning model to classify handwritten digits using the **digits dataset** from scikit-learn. This will help you apply the concepts learned in the session and get hands-on experience with data preprocessing, model training, and evaluation.

---

## Prerequisites: Setting Up Python or Anaconda

### Option 1: Installing Python

1. **Download Python**  
   - Go to the official Python website: [https://www.python.org/downloads/](https://www.python.org/downloads/).  
   - Download the latest version of Python for your operating system (Windows, macOS, or Linux).  
   - Ensure you select the version that matches your OS architecture (32-bit or 64-bit).

2. **Install Python**  
   - Run the downloaded installer.  
   - During installation, **check the box that says "Add Python to PATH"** before proceeding.  
   - Select "Customize Installation" if you'd like to choose advanced settings (optional).  

3. **Verify Installation**  
   - Open a terminal or command prompt.  
   - Type:  
     ```bash
     python --version
     ```  
     or  
     ```bash
     python3 --version
     ```  
     This should display the Python version installed.

4. **Install pip (if not already installed)**  
   Pip is Python's package manager and usually comes pre-installed. Verify by typing:  
   ```bash
   pip --version
   ```  
   If not installed, follow the instructions [here](https://pip.pypa.io/en/stable/installation/).

---

### Option 2: Installing Anaconda (Recommended for Beginners)

1. **Download Anaconda**  
   - Go to [https://www.anaconda.com/](https://www.anaconda.com/).  
   - Download the Anaconda distribution for your operating system.

2. **Install Anaconda**  
   - Run the installer.  
   - Follow the instructions and ensure "Add Anaconda to my PATH environment variable" is checked (optional but recommended for advanced users).  

3. **Verify Installation**  
   - Open the Anaconda Navigator by searching for it in your system applications.  
   - Alternatively, open a terminal or Anaconda Prompt and type:  
     ```bash
     conda --version
     ```  

4. **Set Up a Virtual Environment**  
   - Create an environment for your project:  
     ```bash
     conda create --name ai_model python=3.8
     ```  
   - Activate the environment:  
     ```bash
     conda activate ai_model
     ```  

---

## Installing Required Libraries

Once Python or Anaconda is installed, install the necessary libraries:

```bash
pip install scikit-learn matplotlib pandas jupyter
```

If using Anaconda:

```bash
conda install scikit-learn matplotlib pandas jupyter
```

---

## Steps to Push Code to GitHub (Just incase if you are new to GitHub)

1. **Initialize a Git Repository**:
   - Navigate to your project folder:
     ```bash
     cd /path/to/your/project
     ```
   - Initialize Git:
     ```bash
     git init
     ```

2. **Add and Commit Files**:
   - Add all files to the staging area:
     ```bash
     git add .
     ```
   - Commit your changes:
     ```bash
     git commit -m "Initial commit: My First AI Model"
     ```

3. **Connect to GitHub**:
   - Create a new repository on GitHub.  
   - Copy the repository URL and run:
     ```bash
     git remote add origin <repository-url>
     ```

4. **Push Files**:
   - Push your code to GitHub:
     ```bash
     git branch -M main
     git push -u origin main
     ```

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
   - Once your work is uploaded, share the repository URL with your instructor.URL in this Microsoft Form - [Here](https://forms.office.com/r/MwtiC6U7Ju?origin=lprLink)

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