import pandas as pd
import numpy as np
import pymrmr as mr
import missingno as msno
import featuretools as ft
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance

import shap

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve, validation_curve, KFold, cross_val_score, GridSearchCV


def train_test_split(X, y, size):
    Xa_train, Xa_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=7)
    return Xa_train, Xa_test, y_train, y_test

def model_fit(model, X_train, y_train):
    """Function to fit selected model"""
    model.fit(X_train, y_train)
    return model

def eval_metric(y_test, y_pred):
    """Function to evaluate model performance"""
    score = metrics.r2_score(y_test, y_pred)
    variance = metrics.explained_variance_score(y_test, y_pred)
    mse = metrics.mean_squared_error(y_test, y_pred)
    mae = metrics.mean_absolute_error(y_test, y_pred)
    print(f"The coefficient of determination score for the predicted cost of transaction is {score}")
    print(f"The explained varience score for the predicted cost of transaction is {variance}")
    print(f"The mean square error for the predicted cost of transaction is {mse}")
    print(f"The mean absolute error for the predicted cost of transaction is {mae}")

def line_of_best_fit(y_test, y_pred):
    # Goodness of Fit
    # Visualizing y_test and y_pred
    true = y_test
    pred = y_pred

    # Plot true values
    true_handle = plt.scatter(true, true, alpha=0.6, color='blue', label='true')

    # Reference line
    fit = np.poly1d(np.polyfit(true, true, 1))
    lims = np.linspace(min(true) - 1, max(true) + 1)
    plt.plot(lims, fit(lims), alpha=0.3, color='black')

    # Plot predicted values
    pred_handle = plt.scatter(true, pred, alpha=0.6, color='red', label='predicted')

    # Legend and show
    #plt.figure(figsize=(8, 6))
    plt.legend(handles=[true_handle, pred_handle], loc='upper left')
    plt.show()

def feature_importances(X, model):
    """Function to get feature importance from tree based model"""
    X_variables = X.iloc[:, :]
    dataset_list = list(X_variables.columns)
    importances = list(model.feature_importances_)
    feature_importances = [(feature, round(importance, 3)) for feature, importance in zip(dataset_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=True)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];
    return dataset_list, importances

def feature_importances_viz(X, model):
    dataset_list, importances = feature_importances(X, model)
    plt.figure(figsize=(20, 6))
    # Set the style
    plt.style.use('fivethirtyeight')
    # list of x locations for plotting
    x_values = list(range(len(importances)))
    # Make a bar chart
    plt.bar(x_values, importances, orientation='vertical')
    # Tick labels for x axis
    plt.xticks(x_values, dataset_list, rotation='vertical')
    # Axis labels and title
    plt.ylabel('Importance');
    plt.xlabel('Variable');
    plt.title('Variable Importances for cost of selling a security');

def permutation_importances(model, X, X_test, y_test):
    X_variables = X.iloc[:, :]
    dataset_list = list(X_variables.columns)
    imp = permutation_importance(model, X_test, y_test)
    sorted_idx = imp.importances_mean.argsort()
    plt.figure(figsize=(20, 6))
    plt.barh(dataset_list[sorted_idx], imp.importances_mean[sorted_idx])
    plt.xlabel('Permutation Importance')

