import streamlit as st
import json
from Components.Navbar import Navbar
from Components.Config import Config
from streamlit_lottie import st_lottie


Config()
Navbar()

# Styling
st.markdown("""
    <style>
    
    h1{
        text-transform: uppercase;
        font-weight: bold;
    }
    .main-header {
        text-align: justify; 
        color: darkblue; 
        font-size: 40px;
        margin-bottom: 20px;
    }
    .section-header {
        color: darkblue; 
        font-size: 30px;
        margin-top: 20px;
    }
    .section-content {
        text-align: justify; 
        font-size: 18px;
        margin-bottom: 20px;
    }
    .step-header {
        color: #555555; 
        font-size: 25px;
        margin-top: 20px;
    }
    .step-content {
        text-align: justify; 
        font-size: 18px;
        margin-bottom: 10px;
    }
    .footer {
        text-align: center;
        font-size: 18px;
        margin-top: 70px;
        color: #666666;
        border-top: 1px solid #CCCCCC;
        padding-top: 10px;
    }
    
    .btn-custom {
        background-color: #2f58c1;
        color: white!;
        padding: 15px 25px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 18px;
        font-weight: bold;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 12px;
        transition: all 0.3s;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .btn-custom:hover {
        background-color: #082675;
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
        text-decoration: none;
    }
    .btn-predict {
        background-color: #008CBA;
    }
    .btn-predict:hover {
        background-color: #007B9A;
    }
    .btn-custom, .btn-predict {
        color: white !important;
        text-decoration: none !important;
    </style>
""", unsafe_allow_html=True)

# Title
st.markdown("<h1 class='main-header'>Welcome to Auto Trainer</h1>", unsafe_allow_html=True)

# Subtitle
st.markdown("<h2 class='section-header'>Your Automatic Data Science Assistant</h2>", unsafe_allow_html=True)


# Load and display the image
def load_lottiefiles(filepath: str):
    with open(filepath, 'r') as file:
        return json.load(file)

lottie = load_lottiefiles("assets/data5.json")
lottie1 = load_lottiefiles("assets/data6.json")
lottie2 = load_lottiefiles("assets/data4.json")

col1,col2,col3 = st.columns(3)
with col1:
    st_lottie(
    lottie,
    speed=0.5,
    height=300,
    loop=False,
    quality='high',
    )

with col2:
    st_lottie(
    lottie2,
    speed=0.5,
    height=300,
    loop=False,
    quality='high',
    
    )
with col3:
    st_lottie(
    lottie1,
    height=300,
    loop=True,
    quality='high',
    
    )   
# Create two columns for buttons
col1, col2 = st.columns(2)


with col1:
    st.markdown(
        """
        <a href="/model_build" target="_self" class="btn-custom">
            Train Model
        </a>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <a href="/prediction" target="_self" class="btn-custom">
            Make Predictions
        </a>
        """,
        unsafe_allow_html=True
    )

# Add some information about the app
st.markdown("""
    <h2 class='section-header'>About Auto Trainer</h2>
    <p class='section-content'>
    Auto Trainer is an intuitive platform designed to simplify your machine learning workflow. 
    Whether you're looking to train a new model or make predictions using existing ones, 
    we've got you covered.
    </p>

    <h3 class='step-header'>Features:</h3>
    <ul class='step-content'>
        <li>Easy-to-use interface for model training</li>
        <li>Quick and efficient prediction capabilities</li>
        <li>Support for various types of data and models</li>
        <li>Real-time results and visualizations</li>
    </ul>

    <p class='section-content'>Get started by choosing one of the options above!</p>
""", unsafe_allow_html=True)

# Footer
st.markdown("""
    <div class='footer'>
    Auto Trainer is designed to make machine learning accessible and straightforward for everyone. By using Auto Trainer, you can focus on learning and understanding machine learning concepts without getting bogged down by coding complexities. Auto Trainer: Streamlining Machine Learning For All!
    </div>
   """, unsafe_allow_html=True)