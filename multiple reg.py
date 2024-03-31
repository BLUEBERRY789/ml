import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import Axes3D
# Load the data from a CSV file
file_path = 'book1.csv' # Replace 'your_dataset.csv' with the actual file path
df = pd.read_csv(file_path)
# Extract features (X) and target variable (y)
X = df[['X1', 'X2']].values
y = df['y'].values
# Train a mul􀆟ple linear regression model
model = LinearRegression()
model.fit(X, y)
# Print the coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
# Plot the data points and the plane predicted by the model
fig = plt.figure(figsize=(10, 5))
ax = fig.add_subplot(111, projection='3d')
# Plot the data points
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Data points')
# Plot the plane predicted by the model
x1, x2 = np.meshgrid(np.linspace(X[:, 0].min(), X[:, 0].max(), 10), np.linspace(X[:, 1].min(), X[:, 1].max(), 10))
y_plane = model.intercept_ + model.coef_[0] * x1 + model.coef_[1] * x2
ax.plot_surface(x1, x2, y_plane, alpha=0.5, color='red', label='Regression plane')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('y')
ax.set_title('Mul􀆟ple Linear Regression')
ax.legend()
plt.show()
