import pandas as pd
import os
import sys
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Initialize Navbar and Config
from Components.Navbar import Navbar
from Components.Config import Config

Config()
Navbar()

# Function for normalization
def normalize_data(data, columns, norm_type):
    if norm_type == "Standardization (Z-score)":
        scaler = StandardScaler()
    elif norm_type == "Min-Max Scaling":
        scaler = MinMaxScaler()
    data[columns] = scaler.fit_transform(data[columns])
    return data

# Function to split and save the dataset
def split_and_save_data(data, features, target, test_size):
    X_train, X_test, y_train, y_test = train_test_split(data[features], data[target], test_size=test_size, random_state=42)
    
    # Store in Streamlit's session state
    st.session_state['X_train'] = X_train
    st.session_state['X_test'] = X_test
    st.session_state['y_train'] = y_train
    st.session_state['y_test'] = y_test
    
    # Create and save train and test datasets
    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)
    
    # Save the train and test datasets
    train_filename = "train_dataset.csv"
    test_filename = "test_dataset.csv"
    train_df.to_csv(train_filename, index=False)
    test_df.to_csv(test_filename, index=False)
    
    st.success(f"Train and Test datasets saved as '{train_filename}' and '{test_filename}' respectively.")
    st.download_button(
        label="Download Train Dataset",
        data=train_df.to_csv(index=False),
        file_name=train_filename,
        mime="text/csv",
    )
    st.download_button(
        label="Download Test Dataset",
        data=test_df.to_csv(index=False),
        file_name=test_filename,
        mime="text/csv",
    )

# Function to display data with an option to toggle the percentage of rows displayed
def display_data_with_toggle(data):
    with st.expander("View Data Table", expanded=False):
        view_percentage = st.slider("Select percentage of data to view", 1, 100, 100, key="view_percentage_slider")
        rows_to_show = int(len(data) * view_percentage / 100)
        st.write(f"Showing {view_percentage}% of the data ({rows_to_show} rows)")
        st.dataframe(data.head(rows_to_show))

# Function to download the processed dataset
def download_button(data):
    csv = data.to_csv(index=False)
    st.download_button(
        label="Download Processed Data",
        data=csv,
        file_name="processed_data.csv",
        mime="text/csv",
    )

# Function to save the processed dataset
def saveFile(data):
    directory = os.getcwd()
    file_name = 'processed_data.csv'
    file_path = os.path.join(directory, file_name)
    
    if st.button('Save Processed Data'):
        try:
            data.to_csv(file_path, index=False, encoding='utf8')
            st.success(f'Saved Successfully as {file_name}')
        except Exception as e:
            st.write(e)

# Function to load the uploaded dataset
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            st.success("Data loaded successfully!")
            return data
        except Exception as e:
            st.warning('Error loading data')
            st.write(e)
    return None

# Main App
st.title('Perform Feature Engineering Here')

# Uploading the dataset
uploaded_file = st.file_uploader("Upload your dataset", type=["csv"])
data = load_data(uploaded_file)

if data is not None:
    # Feature Engineering Options
    st.markdown("### Feature Engineering Options")
    
    # Drop columns with 40% or more missing values
    if st.checkbox('Drop columns with 40% or more missing values'):
        missing_threshold = 0.4
        cols_to_drop = data.columns[data.isnull().mean() > missing_threshold]
        data = data.drop(columns=cols_to_drop)
        st.write(f"Columns dropped: {', '.join(cols_to_drop)}")
    
    # Drop columns with 100% unique values
    if st.checkbox('Drop columns with 100% unique values'):
        cols_to_drop = [col for col in data.columns if data[col].nunique() == len(data)]
        data = data.drop(columns=cols_to_drop)
        st.write(f"Columns dropped: {', '.join(cols_to_drop)}")
    
    # Impute missing values
    if st.checkbox('Impute missing values'):
        impute_strategy = st.selectbox("Select imputation strategy", ["Mean (numerical data)", "Mode (categorical data)"])
        if impute_strategy == "Mean (numerical data)":
            imputer = SimpleImputer(strategy='mean')
            numerical_cols = data.select_dtypes(include=['float64', 'int64']).columns
            data[numerical_cols] = imputer.fit_transform(data[numerical_cols])
        elif impute_strategy == "Mode (categorical data)":
            imputer = SimpleImputer(strategy='most_frequent')
            categorical_cols = data.select_dtypes(include=['object']).columns
            data[categorical_cols] = imputer.fit_transform(data[categorical_cols])
        st.write("Missing values imputed successfully.")
    
    # Normalize selected columns
    if st.checkbox('Normalize selected columns'):
        norm_type = st.selectbox("Select normalization type", ["Standardization (Z-score)", "Min-Max Scaling"])
        columns_to_normalize = st.multiselect('Select columns to normalize', data.select_dtypes(include=["float64", "int64"]).columns)
        if columns_to_normalize:
            data = normalize_data(data, columns_to_normalize, norm_type)
            st.write("Selected columns normalized successfully.")
        else:
            st.warning("No columns selected for normalization.")

    display_data_with_toggle(data)
    
    # Encoding options
    encoding_type = st.selectbox('Select your encoding type', ('🏷️Label Encoder', '♨️OneHot Encoder'))
    
    if encoding_type == '🏷️Label Encoder':
        with st.expander('Encoding your data with label encoder'):
            columns_to_encode = st.multiselect('Select the columns you want to encode', data.select_dtypes(include="object").columns)
            if columns_to_encode:
                le = LabelEncoder()
                for column in columns_to_encode:
                    data[column] = le.fit_transform(data[column])
                st.write("Encoding applied successfully.")
                st.table(data)
                download_button(data)
            else:
                st.warning("No columns selected for encoding.")
    elif encoding_type == '♨️OneHot Encoder':
        with st.expander('Encoding your data with one-hot encoding'):
            data = pd.get_dummies(data)
            st.write("One-hot encoding applied successfully.")
            st.table(data)
            download_button(data)
    
    # Split and save datasets
    if st.checkbox("Split and save the dataset"):
        with st.expander("Select Features and Target for Splitting"):
            feature_columns = st.multiselect("Select your Features Columns", data.columns)
            target_column = st.selectbox("Select your labels/Targets Column", data.columns)
        
        if feature_columns and target_column:
            test_size = st.slider("Select test data size", 0.1, 0.5, 0.2)
            split_and_save_data(data, feature_columns, target_column, test_size)
        else:
            st.warning("Please select both features and target columns for splitting.")
else:
    st.warning("Upload a dataset to get started.")
