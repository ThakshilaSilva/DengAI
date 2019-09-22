from __future__ import print_function
from __future__ import division

import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

from matplotlib import pyplot as plt
import seaborn as sn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

import statsmodels.api as sm

from scipy import stats

from sklearn.ensemble import RandomForestRegressor

from warnings import filterwarnings
filterwarnings('ignore')

train_features = pd.read_csv('F:/Sem 7/ML/Project_DengAI/dengue_features_train.csv')
train_labels = pd.read_csv('F:/Sem 7/ML/Project_DengAI/dengue_labels_train.csv')
test_features = pd.read_csv('F:/Sem 7/ML/Project_DengAI/dengue_features_test.csv')

features = ['ndvi_ne', 'ndvi_nw','ndvi_se', 'ndvi_sw', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
       'reanalysis_avg_temp_k', 'reanalysis_dew_point_temp_k',
       'reanalysis_max_air_temp_k', 'reanalysis_min_air_temp_k',
       'reanalysis_precip_amt_kg_per_m2',
       'reanalysis_relative_humidity_percent', 'reanalysis_sat_precip_amt_mm',
       'reanalysis_specific_humidity_g_per_kg', 'reanalysis_tdtr_k',
       'station_avg_temp_c', 'station_diur_temp_rng_c', 'station_max_temp_c',
       'station_min_temp_c', 'station_precip_mm']

test_sj = test_features[test_features['city'] == "sj"]
test_iq = test_features[test_features['city'] == "iq"]

# concatnate train_features with train_labels
train_features = pd.concat([train_labels['total_cases'], train_features], axis=1)

# seperate into two cities
X_sj = train_features[train_features['city'] == "sj"]
X_iq = train_features[train_features['city'] == "iq"]

def fillMissingValues(df, imputer):
    
    imputer.fit(df[features])
    df[features] = imputer.transform(df[features])
    return df

imputer_sj = Imputer(strategy = 'mean')
X_sj = fillMissingValues(X_sj, imputer_sj)
test_sj = fillMissingValues(test_sj, imputer_sj)

imputer_iq = Imputer(strategy = 'mean')
X_iq = fillMissingValues(X_iq, imputer_iq)
test_iq = fillMissingValues(test_iq, imputer_iq)

def transformTemperatureValues(df):
    
    df['reanalysis_air_temp_k'] = df.reanalysis_air_temp_k -273.15
    df['reanalysis_avg_temp_k'] = df.reanalysis_avg_temp_k-273.15
    df['reanalysis_dew_point_temp_k'] = df.reanalysis_dew_point_temp_k-273.15
    df['reanalysis_max_air_temp_k'] = df.reanalysis_max_air_temp_k-273.15
    df['reanalysis_min_air_temp_k'] = df.reanalysis_min_air_temp_k-273.15
    
    return df

X_sj = transformTemperatureValues(X_sj)
X_iq = transformTemperatureValues(X_iq)
test_sj = transformTemperatureValues(test_sj)
test_iq = transformTemperatureValues(test_iq)

importantFeature = ['reanalysis_specific_humidity_g_per_kg', 
                    'reanalysis_dew_point_temp_k', 
                    'reanalysis_min_air_temp_k',
                    'station_min_temp_c',
                    'station_max_temp_c',
                    'station_avg_temp_c']

dropFeatures = list(set(features) - set(importantFeature))

def droppingFeatures(df):
    df.drop(dropFeatures, axis=1, inplace=True)
    return df

X_sj = droppingFeatures(X_sj)
X_iq = droppingFeatures(X_iq)
test_sj = droppingFeatures(test_sj)
test_iq = droppingFeatures(test_iq)

def normalizeData(feature):
    return (feature - feature.mean()) / feature.std()

X_sj[importantFeature] = X_sj[importantFeature].apply(normalizeData, axis=0)
X_iq[importantFeature] = X_iq[importantFeature].apply(normalizeData, axis=0)
test_sj[importantFeature] = test_sj[importantFeature].apply(normalizeData, axis=0)
test_iq[importantFeature] = test_iq[importantFeature].apply(normalizeData, axis=0)

# drop columns
dropping_columns = ['city']
X_sj = X_sj.drop(dropping_columns, axis=1)
X_iq = X_iq.drop(dropping_columns, axis=1)
test_sj = test_sj.drop(dropping_columns, axis=1)
test_iq = test_iq.drop(dropping_columns, axis=1)

# remove outliers
X_sj = X_sj[(np.abs(stats.zscore(X_sj.drop(['year','weekofyear','week_start_date','total_cases'],axis=1))) < 4).all(axis=1)]
X_iq = X_iq[(np.abs(stats.zscore(X_iq.drop(['year','weekofyear','week_start_date','total_cases'],axis=1))) < 4).all(axis=1)]

L_sj = pd.DataFrame(X_sj['total_cases'])
L_iq = pd.DataFrame(X_iq['total_cases'])

# drop total_cases and back X_sj,X_iq in dataset
X_sj = X_sj.drop(['total_cases'],axis=1)
X_iq = X_iq.drop(['total_cases'],axis=1)


sj_correlations = pd.concat([X_sj, L_sj], axis=1).corr().total_cases.drop('total_cases')
iq_correlations = pd.concat([X_iq, L_iq], axis=1).corr().total_cases.drop('total_cases')


for i in X_sj.drop(['year','week_start_date'], axis=1).columns.values:
    X_sj[i] = X_sj[i] * np.absolute(sj_correlations[i]) * 100
    X_iq[i] = X_iq[i] * np.absolute(iq_correlations[i]) * 100

forest_model_sj = RandomForestRegressor(n_estimators=100, max_depth=3, random_state=67)
forest_model_sj.fit(X_sj.drop(['week_start_date','year'], axis=1), L_sj)

forest_model_iq = RandomForestRegressor(n_estimators=180, max_depth=6, random_state=67)
forest_model_iq.fit(X_iq.drop(['week_start_date','year'], axis=1), L_iq)

forest_predict_sj = forest_model_sj.predict(test_sj.drop(['week_start_date','year'], axis=1))
forest_predict_iq = forest_model_iq.predict(test_iq.drop(['week_start_date','year'], axis=1))

predict_list = list((forest_predict_sj).astype(int)) + list((forest_predict_iq).astype(int))

Submission = pd.read_csv('F:/Sem 7/ML/Project_DengAI/submission_format.csv')

Submission['total_cases'] = predict_list

Submission.to_csv('F:/Sem 7/ML/Project_DengAI/randomForestResults.csv', index=False)