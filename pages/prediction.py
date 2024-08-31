#importing packages
import io
import streamlit as st
import pandas as pd
import numpy as np
import pickle as pk
import os
import sys
import time

#import app components
from Components.Navbar import Navbar;
from Components.Config import Config;
from pathlib import Path

Config()
Navbar()

st.markdown("""<h1> Prediction  Area </h1>""", unsafe_allow_html=True)


def uploadModel(filename):
        # load the model from disk
        loaded_model = pk.load(open(filename, 'rb'))
        return loaded_model


file_source = st.selectbox('Choose a file source',('local','Onsite'))
if file_source == 'local':
    file = st.file_uploader("Upload a model file", type="pkl")
    if file is not None:
        filename = file.name
        @st.cache_data
        def getFilePath(filename):
            for root, dirs, files in os.walk(Path.home()):
                for name in files:
                    if name == filename:
                        return os.path.abspath(os.path.join(root, name))
            return None

        with st.spinner(text='Loading model...'):
            time.sleep(1)
            filename = getFilePath(filename=filename)
            st.success('Model loaded')

        # Upload the test dataset
        test_data = st.file_uploader("Upload your test dataset (CSV)", type="csv")
        if test_data is not None:
            test_df = pd.read_csv(test_data)
            st.write("Test dataset preview:")
            st.dataframe(test_df)

            # Select the columns to be used for prediction
            columns = st.multiselect('Select the columns for prediction', test_df.columns)

            # Ensure columns are selected
            if len(columns) > 0:
                # Extract the selected features for prediction
                input_data = test_df[columns].values
            else:
                st.warning("Please select at least one column for prediction.")
                st.stop()
    else:
        st.write('No model file uploaded')
        st.stop()
else:
    available_models = [i for i in os.listdir(os.getcwd()) if i.endswith('.pkl')]
    with st.expander("Select Your Model From The Available Models"):
        selected_model = st.selectbox('Select your model', available_models)
    
    filename = selected_model
    test_data = st.file_uploader("Upload your test dataset (CSV)", type="csv")
    if test_data is not None:
        test_df = pd.read_csv(test_data)
        st.write("Test dataset preview:")
        st.dataframe(test_df)

        columns = st.multiselect('Select the columns for prediction', test_df.columns)

        if len(columns) > 0:
            input_data = test_df[columns].values
        else:
            st.warning("Please select at least one column for prediction.")
            st.stop()

model = uploadModel(filename)
if st.button('Predict'):
    predicted_output = model.predict(input_data)
    st.write('The Predicted output is:')
    st.dataframe(predicted_output)