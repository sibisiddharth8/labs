import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = datasets.load_iris()
X = iris.data[:, :2]  # Using only the first two features for simplicity
y = (iris.target != 0) * 1  # Convert classes to binary (1 if not Iris-setosa, 0 otherwise)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perceptron without bias
perceptron_without_bias = Perceptron(penalty=None, alpha=0.0, fit_intercept=False, max_iter=1000)
perceptron_without_bias.fit(X_train, y_train)

# Perceptron with bias
perceptron_with_bias = Perceptron(penalty=None, alpha=0.0, fit_intercept=True, max_iter=1000)
perceptron_with_bias.fit(X_train, y_train)

# Make predictions
y_pred_without_bias = perceptron_without_bias.predict(X_test)
y_pred_with_bias = perceptron_with_bias.predict(X_test)

# Evaluate accuracy
accuracy_without_bias = accuracy_score(y_test, y_pred_without_bias)
accuracy_with_bias = accuracy_score(y_test, y_pred_with_bias)

# Plot the decision boundaries
def plot_decision_boundary(model, title):
    h = .02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.8)
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, edgecolors='k', marker='o', s=100, linewidth=1, cmap=plt.cm.Paired)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# Plot decision boundaries for perceptrons with and without bias
plot_decision_boundary(perceptron_without_bias, 'Perceptron without Bias')
plot_decision_boundary(perceptron_with_bias, 'Perceptron with Bias')

# Display accuracies
print(f'Accuracy without bias: {accuracy_without_bias:.2f}')
print(f'Accuracy with bias: {accuracy_with_bias:.2f}')
