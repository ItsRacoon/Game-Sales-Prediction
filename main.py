import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = 'vgsales.csv'
data = pd.read_csv("./vgsales.csv")

# Inspect data for preprocessing
print("Data Shape:", data.shape)

# Handle missing values in 'Year' (assuming NaN values indicate missing years)
data['Year'] = data['Year'].fillna(data['Year'].median())

# Selecting relevant features and target
features = ['Platform', 'Year', 'Genre', 'Publisher', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
target = 'Global_Sales'

# Split dataset into features and target variable
X = data[features]
y = data[target]

# Preprocessing for categorical and numerical features
categorical_features = ['Platform', 'Genre', 'Publisher']
numerical_features = ['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']

# Preprocessing pipelines
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the Gradient Boosting model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', GradientBoostingRegressor(random_state=42))
])

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R2 Score: {r2}")

# Plot actual vs predicted sales
plt.figure(figsize=(10, 6))
plt.scatter(y_test, predictions, alpha=0.6, color='b')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Prediction Line')
plt.title('Actual vs Predicted Global Sales')
plt.xlabel('Actual Sales (in millions)')
plt.ylabel('Predicted Sales (in millions)')
plt.legend()
plt.grid(True)
plt.show()

# Example prediction
example_data = pd.DataFrame({
    'Platform': ['Wii'],
    'Year': [2008],
    'Genre': ['Sports'],
    'Publisher': ['Nintendo'],
    'NA_Sales': [15.85],
    'EU_Sales': [12.88],
    'JP_Sales': [3.79],
    'Other_Sales': [3.31]
})
example_prediction = model.predict(example_data)
print(f"Predicted Global Sales for Example Data: {example_prediction[0]} million")





from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# Extracting trees from the GradientBoostingRegressor model
n_trees = len(model.named_steps['regressor'].estimators_)

# Get the feature names after preprocessing
# Apply the column transformer to the data (without actually fitting the model again)
X_transformed = preprocessor.fit_transform(X)

# Get the feature names after one-hot encoding
cat_feature_names = model.named_steps['preprocessor'].transformers_[1][1].named_steps['onehot'].get_feature_names_out(categorical_features)

# Combine numerical feature names with the one-hot encoded categorical feature names
feature_names = numerical_features + list(cat_feature_names)

# Select the first and last tree
tree_1 = model.named_steps['regressor'].estimators_[0, 0]  # First tree
tree_last = model.named_steps['regressor'].estimators_[-1, 0]  # Last tree

# Plot the first tree (no change from before)
plt.figure(figsize=(12, 8))
plt.title('First Tree in Gradient Boosting Model')
plot_tree(tree_1, filled=True, feature_names=feature_names, class_names=["Global_Sales"], rounded=True)


# Plot the last tree with a maximum depth to avoid zoomed-out view
plt.figure(figsize=(12, 8))
plt.title('Last Tree in Gradient Boosting Model')

# Set max_depth to 5 (you can experiment with different values for better readability)
plot_tree(tree_last, filled=True, feature_names=feature_names, class_names=["Global_Sales"], rounded=True, max_depth=5)

plt.show()
