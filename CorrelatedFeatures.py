import pandas as pd
import numpy as np

import shap
import streamlit as st
import streamlit.components.v1 as components
np.random.seed(0)
import matplotlib.pyplot as plt

import lime
import lime.lime_tabular

from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor

st.title("SHAP  & LIME in Analyzer")
st.header("Hafeeza Mughal")

#import data, get first 1000 rows
df = pd.read_csv('winequality-red[1].csv') 
df = df.iloc[:1000]
df['quality'] = df['quality'].astype(int)

#process data
Y = df['quality']
X =  df[['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
       'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
       'pH', 'sulphates', 'alcohol']]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

#train model
model = RandomForestRegressor(max_depth=6, random_state=0, n_estimators=10)
model.fit(X_train, Y_train) 

#LIME analysis
l_explainer = lime.lime_tabular.LimeTabularExplainer(np.array(X_train),
                    feature_names=X.columns, 
                    class_names=['quality'], 
                    verbose=True, mode='regression')

#LIME explanation of first sample
l_exp = l_explainer.explain_instance(X_test.iloc[0], model.predict)
st.subheader('LIME explanation of first sample')
components.html(l_exp.as_html(), width = 800, height = 800, scrolling = True)

#LIME explanation of sencond sample
l_exp = l_explainer.explain_instance(X_test.iloc[1], model.predict)
st.subheader('LIME explanation of second sample')
components.html(l_exp.as_html(), width = 800, height = 800, scrolling = True)

