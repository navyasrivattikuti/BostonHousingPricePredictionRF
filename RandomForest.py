import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = 'c://Users//navya//OneDrive - University of Massachusetts Boston//Documents//Courses//Projects//Git_ML//boston-housing.csv'
housing_data = pd.read_csv(file_path, sep=',', header=0)

# Data preview
print("Data Preview:")
print(housing_data.head())

# Check for NA values
print("\nNA Values:")
print(housing_data.isna().sum())

# Describe the dataset
print("\nDataset Description:")
print(housing_data.describe())

# Data cleaning/preparation
# Check for missing values
housing_data.dropna(inplace=True)

# Feature Engineering: Adding polynomial features and interaction terms
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(housing_data.drop('MEDV', axis=1))
X_poly_df = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(housing_data.drop('MEDV', axis=1).columns))

# Define target variable
y = housing_data['MEDV']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_poly_df, y, test_size=0.2, random_state=42)

# Hyperparameter Tuning: Use GridSearchCV to find the best parameters
param_grid = {
    'n_estimators': [10, 30, 50],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 4, 6],
    'min_samples_leaf': [1, 2, 3],
    'bootstrap': [True, False]
}

rf = RandomForestRegressor(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)

# Best parameters from GridSearchCV
best_params = grid_search.best_params_
print(f"Best parameters: {best_params}")

# Train the Random Forest Regressor with the best parameters
rf_best = RandomForestRegressor(**best_params, random_state=42)
rf_best.fit(X_train, y_train)

# Predict on the test set
y_pred = rf_best.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"R-squared: {r2}")

# Feature importance
feature_importances = rf_best.feature_importances_
features = X_poly_df.columns
feature_importance_df = pd.DataFrame({'Feature': features, 'Importance': feature_importances})
print("\nFeature Importances:")
print(feature_importance_df.sort_values(by='Importance', ascending=False))

# Plotting feature importances (optional)
plt.figure(figsize=(12, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance_df.sort_values(by='Importance', ascending=False).head(20))
plt.title('Top 20 Feature Importances')
plt.show()