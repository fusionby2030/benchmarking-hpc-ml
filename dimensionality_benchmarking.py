"""
func1: variable dataset size (n_features)
    -sklearn.datasets.make_regression

"""

from collections import defaultdict
import time
import gc
import numpy as np
import matplotlib.pyplot as plt


import sklearn
with sklearn.config_context(assume_finite=True):
    pass


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Modesl
from xgboost import XGBRegressor
# import torch


def generate_dataset(n_train: int, n_test: int, n_features: int,  n_targets: int=1):
    X, y = make_regression(n_samples=n_train+n_test+500, n_features=n_features, n_informative=n_features, n_targets=n_targets)
    X_train, X_test, y_train, y_test = train_test_split(X, y,  test_size=n_test, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=n_train, random_state=42)

    X_scaler = StandardScaler()
    X_train = X_scaler.fit_transform(X_train)
    X_val = X_scaler.transform(X_val)
    X_test = X_scaler.transform(X_test)

    y_scaler = StandardScaler()
    y_train = y_scaler.fit_transform(y_train[:, None])[:, 0]
    y_val = y_scaler.transform(y_val[:, None])[:, 0]
    y_test = y_scaler.transform(y_test[:, None])[:, 0]

    gc.collect()

    return X_train, X_val, y_train, y_val, X_test, y_test



def n_feature_influence(estimators, n_train, n_test, n_features, percentile):
    """
    Estimate influence of the number of features on prediction time.

    Parameters
    ----------

    estimators : dict of (name (str), estimator) to benchmark
    n_train : nber of training instances (int)
    n_test : nber of testing instances (int)
    n_features : list of feature-space dimensionality to test (int)
    percentile : percentile at which to measure the speed (int [0-100])

    Returns:
    --------

    percentiles : dict(estimator_name,
                       dict(n_features, percentile_perf_in_us))

    """
    percentiles = defaultdict(defaultdict)
    for n in n_features:
        print("benchmarking with %d features" % n)
        X_train, X_val, y_train, y_val, X_test, y_test = generate_dataset(n_train, n_test, n)
        for cls_name, estimator in estimators.items():
            estimator.fit(X_train, y_train)
            gc.collect()
            runtimes = bulk_benchmark_estimator(estimator, X_test, 30, False)
            percentiles[cls_name][n] = 1e6 * np.percentile(runtimes,
                                                           percentile)
    return percentiles

def plot_n_features_influence(percentiles, percentile):
    fig, ax1 = plt.subplots(figsize=(10, 6))
    colors = ['r', 'g', 'b']
    for i, cls_name in enumerate(percentiles.keys()):
        x = np.array(sorted([n for n in percentiles[cls_name].keys()]))
        y = np.array([percentiles[cls_name][n] for n in x])
        plt.plot(x, y, color=colors[i], )
    ax1.yaxis.grid(True, linestyle='-', which='major', color='lightgrey',
                   alpha=0.5)
    ax1.set_axisbelow(True)
    ax1.set_title('Evolution of Prediction Time with #Features')
    ax1.set_xlabel('#Features')
    ax1.set_ylabel('Prediction Time at %d%%-ile (us)' % percentile)
    plt.show()

# #############################################################################
# Main code

start_time = time.time()

# #############################################################################

configuration = {'n_train': int(1e4),
                  'n_test': int(1e3),
                  'n_features': int(16),
                  'estimators': [
                        {'name': 'XGboost',
                         'instance': XGBRegressor(n_estimators=2000, n_jobs=-1),
                         'complexity_label': 'estimators',
                         'complexity_computer': 2000},

                         ]
}

num_features = [8, 16, 32, 64, 128, 256, 512]
percentile = 90
percentiles = n_feature_influence({'XGboost':  XGBRegressor(n_estimators=2000, n_jobs=-1)},
                                  configuration['n_train'],
                                  configuration['n_test'],
                                  num_features, percentile)
plot_n_features_influence(percentiles, percentile)
