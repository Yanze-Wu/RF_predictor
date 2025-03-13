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
    "apsiii", "temperature", "glucose level", "betablocker", "acei_arb",
    "Vasoactive_drugs", "spo2", "aspirin", "heart rate",
     "loop_diuretics", "cardiogenic shock", "respiratory rate", "age"
]

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# age: numerical input,默认值为 50.0
age = st.number_input("age:", min_value=1, max_value=120, value=50)
# apsiii: numerical input
apsiii = st.number_input("apsiii:", min_value=0, max_value=160, value=50)
# temperature: numerical input
temperature = st.number_input("temperature:", min_value=30.0, max_value=41.0, value=36.8)
# Glu: numerical input,默认值为 50.0
Glu = st.number_input("glucose level:", min_value=0, max_value=1500, value=280)
# betablocker: categorical selection
betablocker = st.selectbox("betablocker:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# acei_arb: categorical selection
acei_arb = st.selectbox("acei_arb:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# Vasoactive_drugs: categorical selection
Vasoactive_drugs = st.selectbox("Vasoactive_drugs:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# spo2: numerical input,默认值为 50.0
spo2 = st.number_input("spo2:", min_value=0, max_value=100, value=98)
# aspirin: categorical selection
aspirin = st.selectbox("aspirin:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# Glu: numerical input,默认值为 50.0
hr = st.number_input("heart rate:", min_value=0, max_value=165, value=98)
# loop_diuretics: categorical selection
loop_diuretics = st.selectbox("loop_diuretics:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# cs: categorical selection
cs = st.selectbox("cardiogenic shock:", options=[0, 1], format_func=lambda x: 'No (0)' if x == 0 else 'Yes (1)')
# rr: numerical input,默认值为 50.0
rr = st.number_input("respiratory rate:", min_value=0, max_value=60, value=18)

# Process inputs and make predictions
feature_values = [age, apsiii, temperature, Glu, betablocker, acei_arb, Vasoactive_drugs, spo2, aspirin, hr, loop_diuretics, cs, rr]
features = np.array([feature_values])


# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[0]
    predicted_proba = model.predict_proba(features)[0]

    # Display prediction results
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")
    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of ICU death. "
            f"The model predicts that your probability of having ICU death is {probability:.1f}%. "
            "While this is just an estimate, it suggests that you may be at significant risk. "
            "I recommend that you consult a cardiologist as soon as possible for further evaluation and "
            "to ensure you receive an accurate diagnosis and necessary treatment."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of ICU death. "
            f"The model predicts that your probability of not having ICU death is {probability:.1f}%. "
            "However, maintaining a healthy lifestyle is still very important. "
            "I recommend regular check-ups to monitor your heart health, "
            "and to seek medical advice promptly if you experience any symptoms."
        )

        st.write(advice)
        # Calculate SHAP values and display force plot
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))
        # 生成力图
        shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)
        plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
        st.image("shap_force_plot.png")
