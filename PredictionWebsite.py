import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

# Title and Description
st.title('Rent Price Prediction for Properties in Brazil')
st.markdown("""
This application allows you to enter the details of a property and get a prediction of the total rent price using a trained Machine Learning model.
""")

# Load the trained model and model columns
best_model = joblib.load('best_model.joblib')
model_columns = joblib.load('model_columns.pkl')

# Sidebar for user input
st.sidebar.header('Property Details')

# Input Fields
area = st.sidebar.number_input('Area (m²)', min_value=10, max_value=1000, value=50)
rooms = st.sidebar.number_input('Number of rooms', min_value=1, max_value=10, value=2)
bathroom = st.sidebar.number_input('Number of bathrooms', min_value=1, max_value=10, value=1)
parking_spaces = st.sidebar.number_input('Parking spaces', min_value=0, max_value=10, value=1)
floor = st.sidebar.number_input('Floor', min_value=0, max_value=50, value=1)
hoa = st.sidebar.number_input('Maintenance fee (hoa) (R$)', min_value=0, value=0)
rent_amount = st.sidebar.number_input('Rent amount (R$)', min_value=0, value=1000)
property_tax = st.sidebar.number_input('Property tax (R$)', min_value=0, value=0)
fire_insurance = st.sidebar.number_input('Fire insurance (R$)', min_value=0, value=0)
city = st.sidebar.selectbox('City', options=['São Paulo', 'Porto Alegre', 'Rio de Janeiro', 'Campinas', 'Belo Horizonte'])
animal = st.sidebar.selectbox('Accepts pets?', options=['aceptan mascotas', 'no aceptan mascotas'])
furniture = st.sidebar.selectbox('Furnished?', options=['amueblado', 'no amueblado'])

# Prediction Button
if st.sidebar.button('Predict Rent Price'):
    # Validate Input
    errors = []
    if area <= 0:
        errors.append('Area must be greater than 0.')
    if rooms <= 0:
        errors.append('Number of rooms must be greater than 0.')
    if bathroom <= 0:
        errors.append('Number of bathrooms must be greater than 0.')
    if rent_amount <= 0:
        errors.append('Rent amount must be greater than 0.')
    
    if errors:
        for error in errors:
            st.error(error)
    else:
        try:
            # Input Data
            data = {
                'area': [area],
                'rooms': [rooms],
                'bathroom': [bathroom],
                'parking spaces': [parking_spaces],
                'floor': [floor],
                'hoa (R$)': [hoa],
                'rent amount (R$)': [rent_amount],
                'property tax (R$)': [property_tax],
                'fire insurance (R$)': [fire_insurance],
                'city': [city],
                'animal': [animal],
                'furniture': [furniture]
            }
            input_df = pd.DataFrame(data)

            # Handling special cases for missing or invalid values
            input_df.replace({'-': np.nan}, inplace=True)

            # Convert categorical variables to dummies
            input_df = pd.get_dummies(input_df, columns=['city', 'animal', 'furniture'])
            
            # Align input columns with model columns
            missing_cols = set(model_columns) - set(input_df.columns)
            for col in missing_cols:
                input_df[col] = 0
            input_df = input_df[model_columns]
            
            # Predict rent price
            prediction = best_model.predict(input_df)
            
            st.success(f'El precio total estimado de alquiler es: R$ {prediction[0]:.2f}')
        except Exception as e:
            st.error(f'An error occurred: {e}')

# Data Exploration Section
st.header('Data Exploration')

# Load CSV
df = pd.read_csv('houses_to_rent_v2.csv')

# Numeric and Categorical Columns
numeric_columns = df.select_dtypes(include=[np.number]).columns
categorical_columns = df.select_dtypes(include=[object]).columns

st.write('Numeric Columns')
st.write(numeric_columns)

st.write('Categorical Columns')
st.write(categorical_columns)

# Data Summary
st.write('Data Summary')
st.write(df.describe())

# Interactive Rental Trends Visualization
st.header('Rental Trends in Different Cities')

# Select city for trends
selected_city = st.selectbox('Select City for Trend Analysis', df['city'].unique())

# Filter data for the selected city
city_data = df[df['city'] == selected_city]

# Interactive Scatter Plot for Rent Trends in the Selected City
fig = px.scatter(city_data, x='area', y='rent amount (R$)', 
                 title=f'Rent Trends in {selected_city}',
                 labels={'area': 'Area (m²)', 'rent amount (R$)': 'Rent Amount (R$)'},
                 trendline="ols")  # Optionally add a trendline to show overall trend
st.plotly_chart(fig)



# Correlation Heatmap (using numeric columns only)
st.write('Correlation Heatmap')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(df[numeric_columns].corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Distribution of Rent Amount
st.write('Distribution of Rent Amount')
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(df['rent amount (R$)'], kde=True, ax=ax)
st.pyplot(fig)

# Boxplot of Rent Amount by City
st.write('Boxplot of Rent Amount by City')
fig, ax = plt.subplots(figsize=(10, 6))
sns.boxplot(x='city', y='rent amount (R$)', data=df, ax=ax)
st.pyplot(fig)

# Count of Properties by City
st.write('Count of Properties by City')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='city', data=df, ax=ax)
st.pyplot(fig)

# Count of Properties by Animal Policy
st.write('Count of Properties by Animal Policy')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='animal', data=df, ax=ax)
st.pyplot(fig)

# Count of Properties by Furniture Policy
st.write('Count of Properties by Furniture Policy')
fig, ax = plt.subplots(figsize=(10, 6))
sns.countplot(x='furniture', data=df, ax=ax)
st.pyplot(fig)