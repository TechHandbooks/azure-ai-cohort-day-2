# Python Scikit Primer 
This example will walk you through creating a basic classifier to predict whether a tumor is malignant or benign using the breast cancer dataset.

# Table of Contents
 - [Steps to Build a Simple ML Model](#steps-to-build-a-simple-ml-model)
   - [1. Set Up the Environment](#1-set-up-the-environment)
   - [2. Import Libraries](#2-import-libraries)
   - [3. Load the Dataset](#3-load-the-dataset)
   - [4. Preprocess the Data](#4-preprocess-the-data)
   - [5. Train the Model](#5-train-the-model)
   - [6. Evaluate the Model](#6-evaluate-the-model)
 - [Full Code Example](#full-code-example)

## Steps to Build a Simple ML Model 
### 1. Set Up the Environment
 
Ensure you have Python installed. You can download it from python.org.
- Install scikit-learn and other necessary libraries using pip:
```
pip install scikit-learn pandas numpy
```

### 2. Import Libraries
 
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### 3. Load the Dataset 
For this example, we’ll use the breast cancer dataset available in scikit-learn.
```
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
```

### 4. Preprocess the Data 
- Split the data into training and testing sets.
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Standardize the features.
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### 5. Train the Model 
Initialize and train a logistic regression model.
```
model = LogisticRegression()
model.fit(X_train, y_train)
```

### 6. Evaluate the Model 
Make predictions and evaluate the accuracy.
Python
```
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

# Full Code Example
[Back to Top](#index)
 
Here’s the complete code for reference:
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_breast_cancer

# Load dataset 
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train model 
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model 
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

