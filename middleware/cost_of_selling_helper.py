import pandas as pd
import numpy as np
import pymrmr as mr
import missingno as msno
import featuretools as ft
import matplotlib.pyplot as plt
import seaborn as sns

import shap

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import f_regression
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import metrics
from xgboost import XGBRegressor
from sklearn.model_selection import ShuffleSplit, train_test_split, learning_curve, validation_curve, KFold, cross_val_score, GridSearchCV


def duplicates(df1,df2):
    """Check for duplicates. Response should be True (no duplicate or False (presence of duplicates)"""
    print(f'Level 1 Data timestamp is unique: {df1.Time_Hour.is_unique}')
    print(f'Level 2 Data timestamp is unique: {df2.Time_Minute.is_unique}')

def missing_stats(df):
    miss_val = df.isna().sum()
    miss_val_per = df.isna().sum() / len(df) * 100
    miss_val_table = pd.concat([miss_val, miss_val_per], axis=1)
    miss_val_table_ren_columns = miss_val_table.rename(columns={0: 'MIssings Values', 1: '% of Total Value'})
    miss_val_table_ren_columns = miss_val_table_ren_columns[miss_val_table_ren_columns.iloc[:, :] != 0].sort_values(
        '% of Total Value', ascending=False).round(1)
    return miss_val_table_ren_columns

def missing_viz_bar(df):
    return msno.bar(df)

def missing_map(df):
    return msno.matrix(df)

def split_time(df):
    """Function to split DateTime to month, day and hour"""
    df['month'] = df['Time_Minute'].dt.month
    df['day'] = df['Time_Minute'].dt.day
    df['hour'] = df['Time_Minute'].dt.hour
    return df

def feature_creation(df):
    """Function to create new features ["count", "mean", "sum", "max", "min", "std"] aggregated to the hour"""
    df1 = df.groupby('Time_Hour_ID').agg(["count", "mean", "sum", "max", "min", "std"])
    columns = []
    # Iterate through the variables name
    for var in df1.columns.levels[0]:
        # skip the id name
        if var != 'Time_Hour_ID':
            # Iterate through the stat names
            for stat in df1.columns.levels[1]:
                # make a new column name for the variable and stat
                columns.append('NEW_%s_%s' % (var, stat))
    df1.columns = columns
    # Remove the columns with all redundant values
    _, idx = np.unique(df1, axis=1, return_index=True)
    df1 = df1.iloc[:, idx]
    return df1

# feature selection
def select_features(X, y, score_funcs, k='all'):
    """Function for feature selection"""
    # configure to select all features
    #f_regression or
    fs = SelectKBest(score_func=score_funcs, k=k)
    # learn relationship from training data
    fs.fit(X, y)
    # transform train input data
    X_fs = fs.transform(X)
    return X_fs, fs

def feature_selction_lr(X, y):
    """feature selection using linear regression"""
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    # define the pipeline to evaluate
    model = LinearRegression()
    fs = SelectKBest(score_func=mutual_info_regression)
    pipeline = Pipeline(steps=[('sel', fs), ('lr', model)])
    # define the grid
    grid = dict()
    grid['sel__k'] = [i for i in range(X.shape[1] - 20, X.shape[1] + 1)]
    # define the grid search
    search = GridSearchCV(pipeline, grid, scoring='neg_mean_squared_error', n_jobs=-1, cv=cv)
    # perform the search
    results = search.fit(X, y)
    # summarize best
    print('Best MAE: %.3f' % results.best_score_)
    print('Best Config: %s' % results.best_params_)
    # summarize all
    means = results.cv_results_['mean_test_score']
    params = results.cv_results_['params']
    for mean, param in zip(means, params):
        print(">%.3f with: %r" % (mean, param))

def train_test_split(X, y, size):
    Xa_train, Xa_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=7)
    return Xa_train, Xa_test, y_train, y_test

def feat_selector(X, y, t, size):
    """Function to carry our feature selection using mRMR
    t = number of features
    X = independent variables
    y= target variable
    size = train-test split ratio
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, size)
    n_features = range(1, t, 1)
    rf_mse = []
    #Instantiate scoring model
    scoring_model = RandomForestRegressor()
    #Instantiate RFCQ mRMR Regressor
    mrmr_regressor_rfcq = MRMRRegressor(relevance='rf')
    for n in n_features:
        print(n)
        sel_features = mrmr_regressor_rfcq.select_best_features(X=X_train,y=y_train, k=n)
        print(n, sel_features)

        #Fit the model
        result = scoring_model.fit(X_train[sel_features], y)
        y_pred_temp = result.predict(X_test[sel_features])
        rf_mse.append(mean_squared_error(y_test, y_pred_temp))


