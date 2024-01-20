import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imblearn.over_sampling import RandomOverSampler
import streamlit as st
import joblib

# 读取训练集数据
train_data = pd.read_csv('train_data - 副本.csv')

# 分离输入特征和目标变量
X = train_data[['Age', 'Sex', 'Histologic', 'Grade', 'T stage', 'N stage',
                'Brain metastasis', 'Liver metastasis', 'Lung metastasis']]
y = train_data['Bone metastasis']

# 创建并训练GBM模型
gbm_model = GradientBoostingClassifier()
gbm_model.fit(X, y)

# 保存模型
model_path = 'gbm_model.pkl'
joblib.dump(gbm_model, model_path)

# 特征映射
feature_order = [
    'Age', 'Sex', 'Histologic', 'Grade', 'T stage', 'N stage',
    'Brain metastasis', 'Liver metastasis', 'Lung metastasis', 'Bone metastasis'
]
class_mapping = {0: "No Bone metastasis", 1: "Esophagus cancer Bone metastasis"}
sex_mapper = {"Female": 2, "Male": 1}
histologic_mapper = {"Adenocarcinoma": 2, "Squamous–cell carcinoma": 1}
grade_mapper = {"Grade I": 3, "Grade II": 1, "Grade III": 2}
t_stage_mapper = {"T1": 4, "T2": 1, "T3": 2, "T4": 3}
n_stage_mapper = {"N0": 4, "N1": 1, "N2": 2, "N3": 3}
brain_metastasis_mapper = {"NO": 0, "Yes": 1}
liver_metastasis_mapper = {"NO": 0, "Yes": 1}
lung_metastasis_mapper = {"NO": 0, "Yes": 1}

# 预测函数
def predict_bone_metastasis(age, sex, histologic, grade,
                            t_stage, n_stage, brain_metastasis, liver_metastasis, lung_metastasis):
    input_data = pd.DataFrame({ 
        'Age': [age],
        'Sex': [sex_mapper[sex]],
        'Histologic': [histologic_mapper[histologic]],
        'Grade': [grade_mapper[grade]],
        'T stage': [t_stage_mapper[t_stage]],
        'N stage': [n_stage_mapper[n_stage]],
        'Brain metastasis': [brain_metastasis_mapper[brain_metastasis]],
        'Liver metastasis': [liver_metastasis_mapper[liver_metastasis]],
        'Lung metastasis': [lung_metastasis_mapper[lung_metastasis]],
    }, columns=feature_order[:-1])

    gbm_model_loaded = joblib.load(model_path)
    prediction = gbm_model_loaded.predict(input_data)[0]
    probability = gbm_model_loaded.predict_proba(input_data)[0][1]  # 获取属于类别1的概率
    class_label = class_mapping[prediction]
    return class_label, probability

# 创建Web应用程序
st.title("GBM Model Predicting Bone Metastasis of Esophageal Cancer")
st.sidebar.write("Variables")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=50)
sex = st.sidebar.selectbox("Sex", options=["Male", "Female"])
histologic = st.sidebar.selectbox("Histologic", options=["Adenocarcinoma", "Squamous–cell carcinoma"])
grade = st.sidebar.selectbox("Grade", options=["Grade I", "Grade II", "Grade III"])
t_stage = st.sidebar.selectbox("T Stage", options=["T1", "T2", "T3", "T4"])
n_stage = st.sidebar.selectbox("N Stage", options=["N0", "N1", "N2", "N3"])
brain_metastasis = st.sidebar.selectbox("Brain Metastasis", options=["NO", "Yes"])
liver_metastasis = st.sidebar.selectbox("Liver Metastasis", options=["NO", "Yes"])
lung_metastasis = st.sidebar.selectbox("Lung Metastasis", options=["NO", "Yes"])

# 预测按钮
if st.button("Predict"):
    prediction, probability = predict_bone_metastasis(age, sex, histologic, grade,
                                                 t_stage, n_stage, brain_metastasis, liver_metastasis, lung_metastasis)

    st.write("Probability of developing bone metastasis: ", prediction)
    st.write("Probability of developing bone metastasis: ", probability)
