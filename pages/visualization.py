import time
import os.path
import numpy as np
import pandas as pd
import seaborn as sns
import sweetviz as sv
import streamlit as st
from io import BytesIO
from csv import DictReader
import plotly.express as px
import matplotlib.pyplot as plt
from Components.Navbar import Navbar;
from Components.Config import Config;
from xml.etree.ElementInclude import include
from sklearn.preprocessing import LabelEncoder, StandardScaler


Config()
Navbar()
#st.markdown("""<h1> Data Visualization Area </h1>""", unsafe_allow_html=True)

def download_button(fig, filename, label):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    st.download_button(
        label=label,
        data=buf,
        file_name=filename,
        mime="image/png"
    )

def load_data():
    if 'data' not in st.session_state:
        st.session_state.data = None
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.session_state.data = data
        return data
    return st.session_state.data

def plot_bar_chart(data, x_column, y_column):
    fig = px.bar(data, x=x_column, y=y_column, title=f'Bar chart of {x_column} vs {y_column}')
    st.plotly_chart(fig)
    buf = BytesIO()
    fig.write_image(buf, format="png")
    st.download_button(
        label=f'Download Bar Chart',
        data=buf.getvalue(),
        file_name=f'bar_chart_{x_column}_vs_{y_column}.png',
        mime="image/png"
    )
    
def plot_histogram(data, column):
    fig, ax = plt.subplots()
    ax.hist(data[column])
    ax.set_title(f'Histogram of {column}')
    ax.set_xlabel(column)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)
    
    download_button(fig, f'histogram_{column}.png', 'Download Histogram')

def plot_scatter(data, x_column, y_column):
    fig = px.scatter(data, x=x_column, y=y_column, title=f'Scatter plot of {x_column} vs {y_column}')
    st.plotly_chart(fig)
    
    buf = BytesIO()
    fig.write_image(buf, format="png")
    st.download_button(
        label=f'Download Scatter Plot',
        data=buf.getvalue(),
        file_name=f'scatter_{x_column}_vs_{y_column}.png',
        mime="image/png"
    )

def plot_boxplot(data, column):
    fig, ax = plt.subplots()
    sns.boxplot(x=data[column], ax=ax)
    ax.set_title(f'Box plot of {column}')
    st.pyplot(fig)
    
    download_button(fig, f'boxplot_{column}.png', 'Download Box Plot')

def plot_correlation_heatmap(data):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Heatmap')
    st.pyplot(fig)
    
    download_button(fig, 'correlation_heatmap.png', 'Download Correlation Heatmap')
    
def plot_pie_chart(data, column):
    fig = px.pie(data, names=column, title=f'Data distribution with Pie Chart')
    st.plotly_chart(fig)

    buf = BytesIO()
    fig.write_image(buf, format="png")
    st.download_button(
        label=f'Download Pie Chart',
        data=buf.getvalue(),
        file_name=f'pie_chart_{column}.png',
        mime="image/png"
    )
    
    
def visualization_page():
    st.title("Data Visualization")

    data = load_data()

    if data is not None:
        st.write(data.head())
        
        # Select encoding type
        encoding_type = st.selectbox('Select your encoding type', ('🏷️Label Encoder', '♨️OneHot Encoder'))
        
        if encoding_type == '🏷️Label Encoder':
            with st.expander('Encoding your data with label encoder'):
                st.markdown('''<p>LabelEncoder can be used to normalize labels. It can also be used to transform non-numerical labels 
                (as long as they are hashable and comparable) to numerical labels.</p>''', unsafe_allow_html=True)

                columns_to_encode = st.multiselect('Select the columns you want to encode', data.select_dtypes(include="object").columns)
                le = LabelEncoder()

                if columns_to_encode:
                    for column in columns_to_encode:
                        data[column] = le.fit_transform(data[column])
                else:
                    st.warning("No columns selected for encoding.")

        elif encoding_type == '♨️OneHot Encoder':
            with st.expander('Encoding your data with one-hot encoding'):
                st.markdown('''<p>One-Hot Encoding is another popular technique for treating categorical variables. It simply creates additional features based on the number of unique values in these
                categorical feature. Every unique value in the category will be added as a feature.</p>''', unsafe_allow_html=True)
                data = pd.get_dummies(data)

        st.subheader("Select Visualization")
        viz_type = st.selectbox("Choose a visualization type", 
                                ["Histogram", "Scatter Plot", "Box Plot", "Correlation Heatmap","Pie Chart","Bar Chart"])

        if viz_type == "Histogram":
            column = st.selectbox("Select a column for the histogram", data.columns)
            plot_histogram(data, column)
            
        elif viz_type == "Bar Chart":
            x_column = st.selectbox("Select the x-axis column", data.columns)
            y_column = st.selectbox("Select the y-axis column", data.columns)
            plot_bar_chart(data, x_column, y_column)
        
        elif viz_type == "Scatter Plot":
            x_column = st.selectbox("Select the x-axis column", data.columns)
            y_column = st.selectbox("Select the y-axis column", data.columns)
            plot_scatter(data, x_column, y_column)
        
        elif viz_type == "Box Plot":
            column = st.selectbox("Select a column for the box plot", data.columns)
            plot_boxplot(data, column)
        
        elif viz_type == "Correlation Heatmap":
            plot_correlation_heatmap(data)
            
        elif viz_type == "Pie Chart":
            column = st.selectbox("Select a column for the pie chart", data.columns)
            plot_pie_chart(data, column)

    else:
        st.write("Please upload a CSV file to visualize data.")

if __name__ == "__main__":
    visualization_page()
