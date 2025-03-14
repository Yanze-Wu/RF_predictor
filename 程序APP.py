import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# Define feature names
feature_names = [
    "age", "temperature", "heart rate", "respiratory rate", "spo2",
    "apsiii", "glucose level", "cardiogenic shock", "acei_arb",
    "aspirin", "betablocker", "loop_diuretics", "Vasoactive_drugs"
]

# Streamlit 界面布局
st.set_page_config(layout="wide")  # 设定宽屏布局

# 左侧：用户输入区域
with st.container():
    st.title("ICU Mortality Prediction with SHAP")

    st.header("Patient Information")
    
    col1, col2, col3 = st.columns(3)  # 创建三列布局

    with col1:
        age = st.number_input("Age:", min_value=1, max_value=120, value=50)
        temperature = st.number_input("Temperature:", min_value=30.0, max_value=41.0, value=36.8)
        hr = st.number_input("Heart Rate:", min_value=0, max_value=165, value=98)
        rr = st.number_input("Respiratory Rate:", min_value=0, max_value=60, value=18)

    with col2:
        spo2 = st.number_input("SpO2:", min_value=0, max_value=100, value=98)
        apsiii = st.number_input("APSI II:", min_value=0, max_value=160, value=50)
        Glu = st.number_input("Glucose Level:", min_value=0, max_value=1500, value=280)
        cs = st.selectbox("Cardiogenic Shock:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

    with col3:
        acei_arb = st.selectbox("ACEI/ARB:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
        aspirin = st.selectbox("Aspirin:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
        betablocker = st.selectbox("Beta Blocker:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
        loop_diuretics = st.selectbox("Loop Diuretics:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
        Vasoactive_drugs = st.selectbox("Vasoactive Drugs:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')

# 组合输入特征
feature_values = [age, temperature, hr, rr, spo2, apsiii, Glu, cs, acei_arb, aspirin, betablocker, loop_diuretics, Vasoactive_drugs]
features = np.array([feature_values])

# 右侧：预测结果和 SHAP 解释
with st.sidebar:
    st.header("Prediction Results")

    if st.button("Predict"):
        # 进行模型预测
        predicted_class = model.predict(features)[0]
        predicted_proba = model.predict_proba(features)[0]

        # 显示预测结果
        st.subheader("Prediction Outcome")
        st.write(f"**Predicted Class:** {'High Risk' if predicted_class == 1 else 'Low Risk'}")
        probability = predicted_proba[predicted_class] * 100
        st.write(f"**Probability:** {probability:.1f}%")

        if predicted_class == 1:
            advice = (
                f"Based on the model's prediction, you have a **high risk of ICU mortality**.\n"
                f"The estimated probability is **{probability:.1f}%**."
            )
        else:
            advice = (
                f"Based on the model's prediction, you have a **low risk of ICU mortality**.\n"
                f"The estimated probability is **{probability:.1f}%**."
            )
        st.info(advice)

        # 计算 SHAP 值并显示解释
        st.subheader("Feature Contribution (SHAP)")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

        # 生成 SHAP 力图
        shap_fig = shap.force_plot(
            explainer.expected_value[predicted_class],
            shap_values[:, :, predicted_class],
            pd.DataFrame([feature_values], columns=feature_names),
            matplotlib=True,
        )
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
