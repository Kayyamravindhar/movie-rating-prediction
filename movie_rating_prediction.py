import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset with cp1252 encoding
data = pd.read_csv('IMDb Movies India.csv', encoding='cp1252')

# Display the first few rows of the dataset
print("Data Head: ")
print(data.head())

# Display the column names to ensure correct selection
print("\nColumns in Dataset: ")
print(data.columns)

# Check for missing values
print("\nMissing Values: ")
print(data.isnull().sum())

# Drop rows with missing values in 'Rating' and 'Votes'
data = data.dropna(subset=['Rating', 'Votes'])

# Convert 'Year' column to numeric (removing parentheses)
data['Year'] = data['Year'].str.extract(r'(\d{4})')  # Extract the year from parentheses
data['Year'] = pd.to_numeric(data['Year'])

# Convert 'Duration' from 'xxx min' to just the number of minutes
data['Duration'] = data['Duration'].str.extract(r'(\d+)').astype(float)

# Remove commas from 'Votes' and convert it to numeric
data['Votes'] = data['Votes'].str.replace(',', '').astype(float)

# Select features and target variable based on available columns
features = ['Year', 'Duration', 'Votes']
X = data[features]
y = data['Rating']

# Impute missing values in X with the mean
imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse}")
print(f"R-Squared Score: {r2}")

# Visualize the results
plt.figure(figsize=(10,6))
sns.scatterplot(x=y_test, y=y_pred, color='blue')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.title('Actual vs Predicted Ratings')
plt.xlabel('Actual Ratings')
plt.ylabel('Predicted Ratings')
plt.show()
