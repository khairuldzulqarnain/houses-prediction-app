import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
    data = pd.read_csv(url)
    return data

data = load_data()

# Display the dataset
st.write("## Housing Dataset")
st.write(data.head())

# Preprocess the data
data = data.dropna(subset=['total_bedrooms'])  # Drop rows with missing values

# Save 'ocean_proximity' values before one-hot encoding
ocean_proximity_values = data['ocean_proximity'].unique()

# One-hot encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Features and target
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# Display the evaluation metrics
st.write("## Model Performance")
st.write(f"Mean Squared Error: {mse}")
st.write(f"Root Mean Squared Error: {rmse}")

# Plot predictions vs actual values
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred, alpha=0.5)
ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
ax.set_xlabel('Actual')
ax.set_ylabel('Predicted')
ax.set_title('Actual vs Predicted')
st.pyplot(fig)

# User input for new prediction
st.write("## Predict New House Price")
st.write("### Enter the details of the house:")
longitude = st.number_input("Longitude", value=data["longitude"].mean())
latitude = st.number_input("Latitude", value=data["latitude"].mean())
housing_median_age = st.number_input("Housing Median Age", value=data["housing_median_age"].mean())
total_rooms = st.number_input("Total Rooms", value=data["total_rooms"].mean())
total_bedrooms = st.number_input("Total Bedrooms", value=data["total_bedrooms"].mean())
population = st.number_input("Population", value=data["population"].mean())
households = st.number_input("Households", value=data["households"].mean())
median_income = st.number_input("Median Income", value=data["median_income"].mean())

# Handle one-hot encoding for the user input
ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_values)
ocean_proximity_dummies = pd.get_dummies(pd.Series([ocean_proximity]), drop_first=True)
ocean_proximity_features = pd.DataFrame(0, index=[0], columns=X.columns[X.columns.str.startswith('ocean_proximity_')])
ocean_proximity_features.update(ocean_proximity_dummies)

# Combine all input features into a single DataFrame
new_data = pd.DataFrame({
    'longitude': [longitude],
    'latitude': [latitude],
    'housing_median_age': [housing_median_age],
    'total_rooms': [total_rooms],
    'total_bedrooms': [total_bedrooms],
    'population': [population],
    'households': [households],
    'median_income': [median_income]
}).join(ocean_proximity_features)

# Reorder columns to match training data
new_data = new_data.reindex(columns=X.columns, fill_value=0)

# Predict the house price based on user input
predicted_price = model.predict(new_data)

st.write(f"## Predicted House Price: ${predicted_price[0]:,.2f}")
