import pandas as pd
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import joblib

# Load the training data
train_data = pd.read_csv('train_data 副本.csv')

# Separate input features and target variable
X = train_data[['Age', 'Sex', 'Tumor histology', 'T stage', 'N stage', 
                'surgery', 'Chemotherapy', 'Brain metastasis', 'Liver metastasis', 'Lung metastasis']]
y = train_data['Bone metastasis']

# Train the LR model
lr_model = LogisticRegression()
lr_model.fit(X, y)

# Feature mapping
sex_mapper = {'male': 1, 'female': 2}
tumor_histology_mapper = {'Adenocarcinoma': 1, 'Squamous–cell carcinoma': 2}
t_stage_mapper = {'T1': 4, 'T2': 1, 'T3': 2, 'T4': 3}
n_stage_mapper = {'N0': 4, 'N1': 1, 'N2': 2, 'N3': 3}
surgery_mapper = {'Yes': 1, 'No': 0}
chemotherapy_mapper = {'Yes': 1, 'No': 0}
brain_metastasis_mapper = {'Yes': 1, 'No': 0}
liver_metastasis_mapper = {'Yes': 1, 'No': 0}
lung_metastasis_mapper = {'Yes': 1, 'No': 0}

# Class label mapping
class_mapping = {0: 'Negative', 1: 'Positive'}

# Prediction function
def predict_Bone_metastasis(age, sex, tumor_histology, 
                             t_stage, n_stage, surgery, chemotherapy, brain_metastasis,
                             liver_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age],
        'Sex': [sex_mapper[sex]],
        'Tumor histology': [tumor_histology_mapper[tumor_histology]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'surgery': [surgery_mapper[surgery]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Liver metastasis': [liver_metastasis_mapper[liver_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = lr_model.predict(input_data)[0]
    probability = lr_model.predict_proba(input_data)[0][1]
    class_label = class_mapping[prediction]
    return class_label, probability

# Create Web application
st.title("LR Model Predicting Bone Metastasis of Esophageal Cancer")
st.sidebar.header("Variables")

# User input interface
age = st.sidebar.slider("Age", min_value=20, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", ('male', 'female'))
tumor_histology = st.sidebar.selectbox("Tumor histology", ('Adenocarcinoma', 'Squamous–cell carcinoma'))
t_stage = st.sidebar.selectbox("T stage", ('T1', 'T2', 'T3', 'T4'))
n_stage = st.sidebar.selectbox("N stage", ('N0', 'N1', 'N2', 'N3'))
surgery = st.sidebar.radio("Surgery", ('Yes', 'No'))
chemotherapy = st.sidebar.radio("Chemotherapy", ('Yes', 'No'))
brain_metastasis = st.sidebar.radio("Brain metastasis", ('Yes', 'No'))
liver_metastasis = st.sidebar.radio("Liver metastasis", ('Yes', 'No'))
lung_metastasis = st.sidebar.radio("Lung metastasis", ('Yes', 'No'))

if st.button("Predict"):
    prediction, probability = predict_Bone_metastasis(age, sex, tumor_histology, 
                                                       t_stage, n_stage, surgery, chemotherapy, 
                                                       brain_metastasis, liver_metastasis, lung_metastasis)

    st.write("Prediction:", prediction)
    st.write("Probability of developing Bone metastasis:", probability)
