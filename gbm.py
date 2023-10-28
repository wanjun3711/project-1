import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import joblib

# 加载模型
model_path = 'gbm_model.model'
gbm_model = GradientBoostingClassifier()
gbm_model = joblib.load(model_path)

# 特征映射
class_mapping = {0: "No liver metastasis", 1: "Esophagus cancer liver metastasis"}
age_mapper = {"<70": 3, "70-80": 2, ">=80": 1}
primary_site_mapper = {"Upper third of esophagus": 4,"Middle third of esophagus": 2,
    "Lower third of esophagus": 3, "Overlapping lesion of esophagus": 1}

histologic_mapper = {"Adenocarcinoma": 2, "Squamous–cell carcinoma": 1}
tumor_grade_mapper = {"Grade I": 3, "Grade II": 2, "Grade III": 1}
t_stage_mapper = {"T1": 4, "T2": 2, "T3": 3, "T4": 1}
n_stage_mapper = {"N0": 4, "N1": 2, "N2": 3, "N3": 1}
surgery_mapper = {"NO": 2, "Yes": 1}
radiation_mapper = {"NO": 2, "Yes": 1}
chemotherapy_mapper = {"NO": 2, "Yes": 1}
bone_metastasis_mapper = {"NO": 2, "Yes": 1}
lung_metastasis_mapper = {"NO": 2, "Yes": 1}

# 预测函数
def predict_liver_metastasis(age, primary_site, histologic, tumor_grade,
                             t_stage, n_stage, surgery, radiation,
                             chemotherapy, bone_metastasis, lung_metastasis):
    input_data = pd.DataFrame({
        'Age': [age_mapper[age]],
        'Primary Site': [primary_site_mapper[primary_site]],
        'Histologic': [histologic_mapper[histologic]],
        'Tumor grade': [tumor_grade_mapper[tumor_grade]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Surgery': [surgery_mapper[surgery]],
        'Radiation': [radiation_mapper[radiation]],
        'Chemotherapy': [chemotherapy_mapper[chemotherapy]],
        'Bone metastasis': [bone_metastasis_mapper[bone_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]]
    })
    prediction = gbm_model.predict(input_data)[0]
    probability = gbm_model.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability
# 创建Web应用程序
st.title("GBM Model Predicting Liver Metastasis of Esophageal Cancer")
st.sidebar.write("Variables")

age = st.sidebar.selectbox("Age", options=list(age_mapper.keys()))
primary_site = st.sidebar.selectbox("Primary site", options=list(primary_site_mapper.keys()))
histologic = st.sidebar.selectbox("Histologic", options=list(histologic_mapper.keys()))
tumor_grade = st.sidebar.selectbox("Tumor grade", options=list(tumor_grade_mapper.keys()))
t_stage = st.sidebar.selectbox("T stage", options=list(t_stage_mapper.keys()))
n_stage = st.sidebar.selectbox("N stage", options=list(n_stage_mapper.keys()))
surgery = st.sidebar.selectbox("Surgery", options=list(surgery_mapper.keys()))
radiation = st.sidebar.selectbox("Radiation", options=list(radiation_mapper.keys()))
chemotherapy = st.sidebar.selectbox("Chemotherapy", options=list(chemotherapy_mapper.keys()))
bone_metastasis = st.sidebar.selectbox("Bone metastasis", options=list(bone_metastasis_mapper.keys()))
lung_metastasis = st.sidebar.selectbox("Lung metastasis", options=list(lung_metastasis_mapper.keys()))

if st.button("Predict"):
    prediction, probability = predict_liver_metastasis(age, primary_site, histologic, tumor_grade,
                                          t_stage, n_stage, surgery, radiation,
                                          chemotherapy, bone_metastasis, lung_metastasis)

    st.write("Probability of developing liver metastasis：", prediction)  # 结果显示在右侧的列中
    st.write("Probability of developing liver metastasis：", probability)  # 结果显示在右侧的列中
