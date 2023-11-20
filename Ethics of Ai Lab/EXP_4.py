import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from mlxtend.plotting import plot_decision_regions
from sklearn.datasets import load_iris  # Import Iris dataset from scikit-learn

# Load the Iris dataset from scikit-learn
iris = load_iris()
X = iris.data[:, [0, 2]]  # Select two features for visualization (you can change this)
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create a perceptron model without bias (no intercept)
perceptron_no_bias = Perceptron(fit_intercept=False, max_iter=1000, random_state=0)
perceptron_no_bias.fit(X_train, y_train)

# Create a perceptron model with bias (intercept)
perceptron_with_bias = Perceptron(fit_intercept=True, max_iter=1000, random_state=0)
perceptron_with_bias.fit(X_train, y_train)

# Make predictions
y_pred_no_bias = perceptron_no_bias.predict(X_test)
y_pred_with_bias = perceptron_with_bias.predict(X_test)

# Calculate accuracy
accuracy_no_bias = accuracy_score(y_test, y_pred_no_bias)
accuracy_with_bias = accuracy_score(y_test, y_pred_with_bias)

print("Accuracy Without Bias (No Intercept):", accuracy_no_bias)
print("Accuracy With Bias (Intercept):", accuracy_with_bias)

# Plot decision boundaries for both models
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plot_decision_regions(X_train, y_train, clf=perceptron_no_bias, legend=2)
plt.xlabel(iris.feature_names[0])  # Update with the appropriate feature name
plt.ylabel(iris.feature_names[2])  # Update with the appropriate feature name
plt.title('Perceptron Without Bias (No Intercept)')

plt.subplot(1, 2, 2)
plot_decision_regions(X_train, y_train, clf=perceptron_with_bias, legend=2)
plt.xlabel(iris.feature_names[0])  # Update with the appropriate feature name
plt.ylabel(iris.feature_names[2])  # Update with the appropriate feature name
plt.title('Perceptron With Bias (Intercept)')

plt.tight_layout()
plt.show()
