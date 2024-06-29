import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Reading data from csv
housing_data=pd.read_csv('c://Users//navya//OneDrive - University of Massachusetts Boston//Documents//Courses//Projects//Git_ML//boston-housing.csv', sep=',')
print("Data Preview:\n")
print(housing_data)

# Check for NA values
print("\nNA Values:")
print(housing_data.isna().sum())

# Describe the dataset
print("\nDataset Description:")
print(housing_data.describe())

# Data cleaning/preparation
housing_data.dropna(inplace=True)

# Define features and target variable
X = housing_data.drop(['MEDV'], axis=1)
y = housing_data['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
rf.fit(X_train, y_train)

# Predict on the test set
y_pred = rf.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Feature importance
feature_importances = rf.feature_importances_
features = X.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("\nFeature Importances:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Plotting feature importances

plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False))
plt.title('Feature Importances')
plt.show()