# Import necessary packages
import streamlit as st
import pandas as pd
import time
import joblib
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error
from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, AgGridTheme

def buildInterractiveTable(data):
    _select = st.sidebar.radio('select', ('Highlight', 'Hide'))
    gd = GridOptionsBuilder.from_dataframe(data)
    gd.configure_pagination(enabled=True)
    gd.configure_default_column(editable=True, groupable=True, resizable=True)

    cel_mode = st.radio('Selection mode', options=['single', 'multiple'])
    gd.configure_selection(selection_mode=cel_mode, use_checkbox=True)
    gridoptions = gd.build()
    gridtable = AgGrid(data, gridOptions=gridoptions, update_mode=GridUpdateMode.VALUE_CHANGED,
                       height=500, allow_unsafe_jscode=True, theme=AgGridTheme.STREAMLIT)
    selected_rows = gridtable['selected_rows']
    return gridtable

class Modelbuilding:
    @staticmethod
    def randomForestClassifier(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                rdm = RandomForestClassifier(
                    n_estimators=params['n_estimators'],
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'],
                    random_state=params['random_state']
                )
                clf = rdm.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def decisionTreeClassifier(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                dtc = DecisionTreeClassifier(
                    splitter=params['splitter'], max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'], min_samples_leaf=params['min_samples_leaf'],
                    max_features=params['max_features'], random_state=params['random_state'],
                    max_leaf_nodes=params['max_leaf_nodes']
                )
                clf = dtc.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error("Select your features or check your parameters")
            st.write(e)
            st.stop()

    @staticmethod
    def knearstNeighborClassifier(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                knn = KNeighborsClassifier(
                    n_neighbors=params['n_neighbors'], weights=params['weights'],
                    algorithm=params['algorithm'], leaf_size=params['leaf_size']
                )
                clf = knn.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def supportVectorMachine(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                svm = SVC(
                    C=params['C'], kernel=params['kernel'], gamma=params['gamma'], random_state=params['random_state']
                )
                clf = svm.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def linearRegression(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                lr = LinearRegression(fit_intercept=params['fit_intercept'])
                clf = lr.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The mean squared error is", mean_squared_error(y_test, predicted_output))
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def logisticRegression(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                lr = LogisticRegression(
                    penalty=params['penalty'], C=params['C'], solver=params['solver'],
                    max_iter=params['max_iter'], random_state=params['random_state']
                )
                clf = lr.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def kMeansClustering(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                km = KMeans(
                    n_clusters=params['n_clusters'], init=params['init'], max_iter=params['max_iter'],
                    n_init=params['n_init'], random_state=params['random_state']
                )
                clf = km.fit(X_train)
                predicted_output = clf.predict(X_test)
                st.header('The Cluster assignments are')
                st.write(predicted_output)
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def naiveBayesClassifier(features, params, X_train, X_test, y_train, y_test):
        try:
            with st.spinner('Wait while your model is being built...'):
                nb = GaussianNB()
                clf = nb.fit(X_train, y_train)
                predicted_output = clf.predict(X_test)
                st.header('The Predicted values are')
                st.write("The accuracy score is", accuracy_score(y_test, predicted_output) * 100, "%")
                return clf
        except Exception as e:
            st.error('Select your features or check your parameters')
            st.write(e)
            st.stop()

    @staticmethod
    def save_model(model):
        file_name = st.text_input("Enter a name for the saved model", "model.pkl")
        if st.button("Save Model"):
            joblib.dump(model, file_name)
            st.success(f"Model saved as {file_name}")
