import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset (replace 'your_data.csv' with your dataset file)
data = pd.read_csv(r'C:\\Users\SRINIVASAN\Desktop\Ethics of Ai Lab\Your_Data.csv')

# Step 2: Summary Statistics
summary_stats = data.describe()
print(summary_stats)

# Step 3: Data Visualization
plt.figure(figsize=(12, 6))

# Scatter plot
plt.subplot(1, 2, 1)
plt.scatter(data['X'], data['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Scatter Plot of X vs. Y')

# Histograms or density plots
plt.subplot(1, 2, 2)
sns.histplot(data['X'], kde=True, label='X')
sns.histplot(data['Y'], kde=True, label='Y')
plt.xlabel('Value')
plt.title('Histograms of X and Y')
plt.legend()

plt.show()

# Step 4: Correlation Analysis
correlation_coefficient = data['X'].corr(data['Y'])
print(f'Correlation Coefficient: {correlation_coefficient:.2f}')

# Correlation matrix and heatmap
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Step 5: Outlier Detection and Handling (if needed)

# Step 6: Data Preprocessing (if needed)

# Step 7: Model Building
X = data[['X']]  # Independent variable
y = data['Y']    # Dependent variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# Plot the regression line
plt.scatter(X_test, y_test, label='Actual')
plt.plot(X_test, y_pred, color='red', label='Predicted')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Linear Regression Model')
plt.legend()
plt.show()
