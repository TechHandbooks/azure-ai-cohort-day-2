from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# Load the Iris dataset
iris = load_iris()

# Dataset details
print("Feature names:", iris.feature_names)
print("Target names:", iris.target_names)

# Display first 5 samples
print("Sample data:", iris.data[:5])
print("Target labels:", iris.target[:5])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.3, random_state=42)

print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

# Initialize the model
model = RandomForestClassifier(random_state=42)

# Train the model
model.fit(X_train, y_train)
print("Model trained successfully!")

# Predict on the test set
y_pred = model.predict(X_test)

print("Predictions:", y_pred)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

