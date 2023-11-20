import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Generate sample data
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 3, 4, 5, 6])

# Create a linear regression model without bias (no intercept)
model_no_bias = LinearRegression(fit_intercept=False)
model_no_bias.fit(X, y)

# Create a linear regression model with bias (intercept)
model_with_bias = LinearRegression(fit_intercept=True)
model_with_bias.fit(X, y)

# Generate predictions
X_pred = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y_pred_no_bias = model_no_bias.predict(X_pred)
y_pred_with_bias = model_with_bias.predict(X_pred)

# Plot the data and regression lines
plt.scatter(X, y, label='Data')
plt.plot(X_pred, y_pred_no_bias, label='Regression Without Bias', linestyle='--', color='r')
plt.plot(X_pred, y_pred_with_bias, label='Regression With Bias', linestyle='--', color='g')

plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('Linear Regression with and without Bias')
plt.grid(True)
plt.show()

# Print coefficients and intercepts
print("Model Without Bias (No Intercept)")
print("Coefficient:", model_no_bias.coef_)

print("\nModel With Bias (Intercept)")
print("Coefficient:", model_with_bias.coef_)
print("Intercept:", model_with_bias.intercept_)
