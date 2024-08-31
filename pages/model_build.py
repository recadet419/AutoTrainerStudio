import os
import pickle
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from Components.functions import Modelbuilding
from Components.Config import Config
from Components.Navbar import Navbar

Config()
Navbar()


st.title('Build your Model Here')

# Read the dataset
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Preprocess the dataset
    st.subheader("Preprocessing")

    # Handle missing values
    if st.checkbox("Fill missing values with median", value=True):
        # Fill missing values for numeric columns only
        numeric_cols = df.select_dtypes(include=['number']).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    else:
        if st.checkbox("Drop rows with missing values"):
            df.dropna(inplace=True)

    # Encode categorical features
    categorical_cols = df.select_dtypes(include=['object']).columns
    if st.checkbox("Encode categorical features"):
        le = LabelEncoder()
        for col in categorical_cols:
            df[col] = le.fit_transform(df[col].astype(str))

            
    st.write("Preprocessed Dataset")
    st.write(df.head())

   
    # Select feature columns and target column
    with st.expander("Select Your Features and Labels"):
            st.markdown(
                    """
                Features are the independent variables for building your model and the
                Targets are the dependent variable, ie the variables to be predicted """,unsafe_allow_html=True
                )
            feature_columns = st.multiselect("Select your Features Columns",df.columns,help='Select the columns you want to use as features')
            target_column = st.selectbox("Select your labels/Targets Columns",df.columns,help='Select the columns you want to use as labels')

    if feature_columns and target_column:
        features = df[feature_columns].to_numpy()
        labels = df[target_column].to_numpy()        
                
        # Select the model
        model_choice = [
            "Random Forest",
            "Decision Tree",
            "K-Nearest Neighbor",
            "SVM",
            "Linear Regression",
            "Logistic Regression",
            "K-Means Clustering",
            "Naive Bayes"
        ]
        
        algorithm = st.radio('Select your preferred algorithm', model_choice)
        st.write('<style>div.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html = True)

        # Parameters section
        param_options = []
        if algorithm == "Random Forest":
            param_options = [
                'n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features'
            ]
        elif algorithm == "Decision Tree":
            param_options = [
                'splitter', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'max_features', 'max_leaf_nodes'
            ]
        elif algorithm == "K-Nearest Neighbor":
            param_options = [
                'n_neighbors', 'weights', 'algorithm', 'leaf_size'
            ]
        elif algorithm == "SVM":
            param_options = [
                'C', 'kernel', 'gamma'
            ]
        elif algorithm == "Linear Regression":
            param_options = [
                'fit_intercept'
            ]
        elif algorithm == "Logistic Regression":
            param_options = [
                'penalty', 'C', 'solver', 'max_iter'
            ]
        elif algorithm == "K-Means Clustering":
            param_options = [
                'n_clusters', 'init', 'max_iter', 'n_init'
            ]

        chosen_params = st.multiselect("Choose Parameters", param_options)

        params = {}
        if 'n_estimators' in chosen_params:
            params['n_estimators'] = st.slider("Number of trees", 1, 200, 100)
        if 'max_depth' in chosen_params:
            params['max_depth'] = st.slider("Maximum depth of the tree", 1, 20, 10)
        if 'min_samples_split' in chosen_params:
            params['min_samples_split'] = st.slider("Minimum number of samples required to split an internal node", 2, 10, 2)
        if 'min_samples_leaf' in chosen_params:
            params['min_samples_leaf'] = st.slider("Minimum number of samples required to be at a leaf node", 1, 10, 1)
        if 'max_features' in chosen_params:
            params['max_features'] = st.radio("Number of features to consider when looking for the best split", ["sqrt", "log2", None])
        if 'splitter' in chosen_params:
            params['splitter'] = st.radio("Strategy used to choose the split at each node", ["best", "random"])
        if 'max_leaf_nodes' in chosen_params:
            params['max_leaf_nodes'] = st.slider("Grow a tree with max_leaf_nodes in best-first fashion", 2, 10, None)
        if 'n_neighbors' in chosen_params:
            params['n_neighbors'] = st.slider("Number of neighbors to use", 1, 20, 5)
        if 'weights' in chosen_params:
            params['weights'] = st.radio("Weight function used in prediction", ["uniform", "distance"])
        if 'algorithm' in chosen_params:
            params['algorithm'] = st.radio("Algorithm used to compute the nearest neighbors", ["auto", "ball_tree", "kd_tree", "brute"])
        if 'leaf_size' in chosen_params:
            params['leaf_size'] = st.slider("Leaf size passed to BallTree or KDTree", 1, 50, 30)
        if 'C' in chosen_params:
            params['C'] = st.slider("Regularization parameter", 0.01, 10.0, 1.0)
        if 'kernel' in chosen_params:
            params['kernel'] = st.radio("Kernel type to be used in the algorithm", ["linear", "poly", "rbf", "sigmoid"])
        if 'gamma' in chosen_params:
            params['gamma'] = st.radio("Kernel coefficient", ["scale", "auto"])
        if 'fit_intercept' in chosen_params:
            params['fit_intercept'] = st.radio("Whether to calculate the intercept for this model", [True, False])
        if 'penalty' in chosen_params:
            params['penalty'] = st.radio("Norm used in the penalization", ["l1", "l2", "elasticnet", "none"])
        if 'solver' in chosen_params:
            params['solver'] = st.radio("Algorithm to use in the optimization problem", ["newton-cg", "lbfgs", "liblinear", "sag", "saga"])
        if 'max_iter' in chosen_params:
            params['max_iter'] = st.slider("Maximum number of iterations taken for the solvers to converge", 50, 300, 100)
        if 'n_clusters' in chosen_params:
            params['n_clusters'] = st.slider("The number of clusters to form", 1, 10, 5)
        if 'init' in chosen_params:
            params['init'] = st.radio("Method for initialization", ["k-means++", "random"])
        if 'n_init' in chosen_params:
            params['n_init'] = st.slider("Number of time the k-means algorithm will be run with different centroid seeds", 1, 20, 10)

        params['random_state'] = 42

        # Default parameters
        params.setdefault('n_estimators', 100)
        params.setdefault('max_depth', None)
        params.setdefault('min_samples_split', 2)
        params.setdefault('min_samples_leaf', 1)
        params.setdefault('max_features', 'auto')
        params.setdefault('splitter', 'best')
        params.setdefault('max_leaf_nodes', None)
        params.setdefault('n_neighbors', 5)
        params.setdefault('weights', 'uniform')
        params.setdefault('algorithm', 'auto')
        params.setdefault('leaf_size', 30)
        params.setdefault('C', 1.0)
        params.setdefault('kernel', 'rbf')
        params.setdefault('gamma', 'scale')
        
        
        # if 'model' not in st.session_state:
        #     st.session_state.model = None
            
        if st.button("Train Model"):
            model = None  # Initialize model to None

            try:
                if 'X_train' in st.session_state and 'y_train' in st.session_state:
                    # Retrieve from session state
                    X_train = st.session_state.X_train
                    X_test = st.session_state.X_test
                    y_train = st.session_state.y_train
                    y_test = st.session_state.y_test

                if algorithm == "Random Forest":
                    model = Modelbuilding.randomForestClassifier(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "Decision Tree":
                    model = Modelbuilding.decisionTreeClassifier(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "K-Nearest Neighbor":
                    model = Modelbuilding.knearstNeighborClassifier(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "SVM":
                    model = Modelbuilding.supportVectorMachine(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "Linear Regression":
                    model = Modelbuilding.linearRegression(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "Logistic Regression":
                    model = Modelbuilding.logisticRegression(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "K-Means Clustering":
                    model = Modelbuilding.kMeansClustering(features, params, X_train, X_test, y_train, y_test)
                elif algorithm == "Naive Bayes":
                    model = Modelbuilding.naiveBayesClassifier(features, labels, params, X_train, X_test, y_train, y_test)
                else:
                    st.error("Unrecognized algorithm. Please check the algorithm name.")

                if model is not None:
                    st.session_state.model = model
                    st.success("Model trained successfully!")
        
            except Exception as e:
                    st.error(f"An error occurred: {e}")

if 'model' in st.session_state and st.session_state.model is not None:
            if st.button("Save Model"):
                model = st.session_state.model
                model_filename = f"{algorithm.replace(' ', '_').lower()}_model.pkl"
                with open(model_filename, "wb") as f:
                    pickle.dump(model, f)
                st.success(f"Model saved as {model_filename}")
                    
                # Provide the download link
                with open(model_filename, "rb") as f:
                    st.download_button(
                        label="Download trained model",
                        data=f,
                        file_name=model_filename,
                        mime='application/octet-stream'
                        )
else:
    st.write("Train a model before saving.")
        