import pandas as pd
import numpy as np

import shap
import streamlit as st
import streamlit.components.v1 as components
import xgboost

import lime
from lime import lime_tabular

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier

##############################################################################

@st.cache
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/dsrscientist/dataset1/master/census_income.csv")
    df.drop(df[df['Native_country'] == ' ?'].index,inplace=True)
    df.drop(df[df['Occupation'] == ' ?'].index,inplace=True)
    le = LabelEncoder() # label encoder 
    df['Income'] = le.fit_transform(df['Income'])
    df['Sex'] = le.fit_transform(df['Sex'])
    df = df.drop(["Education", "Workclass", "Marital_status", "Race", "Native_country", "Relationship", "Occupation"], axis=1)

    return df
    #return shap.datasets.boston()

def st_shap(plot, height=400, width = 400, scrolling = True):
    shap_html = f"<head>{shap.getjs()}</head><body>{plot.html()}</body>"
    components.html(shap_html, width = width, height = height, scrolling = True)

    
df = load_data()
X = df.drop(['Income'], axis=1)
s = MinMaxScaler()
X[X.columns] = s.fit_transform(X[X.columns])
y = df['Income']

st.title("SHAP  & LIME in Analyzer")
st.subheader("Hafeeza Mughal")


# train  model
##model = xgboost.train({"learning_rate": 0.01}, xgboost.DMatrix(X, label=y), 100)
X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.33, random_state=2022)
clf = MLPClassifier(max_iter=100, random_state=2022)
clf.fit(X_train, y_train)

#explain and visualize with LIME
lime_explainer = lime_tabular.LimeTabularExplainer(training_data = np.array(X_train),
                                              feature_names = X_train.columns,
                                              class_names = ['0','1'],
                                              mode = 'classification')

lime_explanation = lime_explainer.explain_instance(data_row = X_test.iloc[2],
                                         predict_fn = clf.predict_proba, 
                                         num_features = 3, 
                                         num_samples = 5)

components.html(lime_explanation.as_html(), width = 800, height = 800, scrolling = True)


# explain the model's predictions using SHAP
shap_explainer = shap.KernelExplainer(clf.predict_proba,X_train)
shap_values = shap_explainer.shap_values(X_test.iloc[0:5,:])

# visualize predictions explanation
st_shap(shap.force_plot(shap_explainer.expected_value[0],shap_values[0], X_test.iloc[0:5,:]))
