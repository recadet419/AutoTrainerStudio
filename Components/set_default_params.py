def default_params(params):
    # Random Forest default parameters
    params.setdefault('n_estimators', 100)
    params.setdefault('max_depth', None)
    params.setdefault('min_samples_split', 2)
    params.setdefault('min_samples_leaf', 1)
    params.setdefault('max_features', 'auto')

    # Decision Tree default parameters
    params.setdefault('splitter', 'best')
    params.setdefault('max_depth', None)
    params.setdefault('min_samples_split', 2)
    params.setdefault('min_samples_leaf', 1)
    params.setdefault('max_features', None)
    params.setdefault('max_leaf_nodes', None)

    # K-Nearest Neighbor default parameters
    params.setdefault('n_neighbors', 5)
    params.setdefault('weights', 'uniform')
    params.setdefault('algorithm', 'auto')
    params.setdefault('leaf_size', 30)

    # SVM default parameters
    params.setdefault('C', 1.0)
    params.setdefault('kernel', 'rbf')
    params.setdefault('gamma', 'scale')

    # Linear Regression default parameters
    params.setdefault('fit_intercept', True)

    # Logistic Regression default parameters
    params.setdefault('penalty', 'l2')
    params.setdefault('C', 1.0)
    params.setdefault('solver', 'lbfgs')
    params.setdefault('max_iter', 100)

    # K-Means Clustering default parameters
    params.setdefault('n_clusters', 8)
    params.setdefault('init', 'k-means++')
    params.setdefault('max_iter', 300)
    params.setdefault('n_init', 10)