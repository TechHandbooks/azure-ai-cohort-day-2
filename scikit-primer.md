# Python Scikit Primer
This example will walk you through creating a basic classifier to predict whether a tumor is malignant or benign using the breast cancer dataset.

## Steps to Build a Simple ML Model
### Set Up the Environment
Ensure you have Python installed. You can download it from python.org.
- Install scikit-learn and other necessary libraries using pip:
```
pip install scikit-learn pandas numpy
```

### Import Libraries
Python
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
```

### Load the Dataset
For this example, we’ll use the breast cancer dataset available in scikit-learn.
Python
```
from sklearn.datasets import load_breast_cancer
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)
```

### Preprocess the Data
- Split the data into training and testing sets.
Python
```
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
- Standardize the features.
Python
```
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
```

### Train the Model
Initialize and train a logistic regression model.
Python
```
model = LogisticRegression()
model.fit(X_train, y_train)
```

### Evaluate the Model
Make predictions and evaluate the accuracy.
Python
```
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy * 100:.2f}%')
```

# Full Code Example
Here’s the complete code for reference:
Python
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
