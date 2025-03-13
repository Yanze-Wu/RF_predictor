import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载保存的随机森林模型
model = joblib.load('RF.pkl')

# 特征范围定义（根据提供的特征范围和数据类型）
feature_ranges = {
    "age": {"type": "numerical", "min": 1.0, "max": 120.0, "default": 50.0},
    "apsiii": {"type": "numerical", "min": 0.0, "max": 160.0, "default": 50.0},
    "temperature": {"type": "numerical", "min": 30.0, "max": 41.0, "default": 36.8},
    "Glu": {"type": "numerical", "min": 0.0, "max": 1500.0, "default": 240.0},
    "betablocker": {"type": "categorical", "options": [0, 1]},
    "acei.arb": {"type": "categorical", "options": [0, 1]},
    "Vasoactive_drugs": {"type": "categorical", "options": [0, 1]},
    "spo2": {"type": "numerical", "min": 0.0, "max": 100.0, "default": 98.0},
    "aspirin": {"type": "categorical", "options": [0, 1]},
    "hr": {"type": "numerical", "min": 0.0, "max": 165.0, "default": 98.0},
    "loop_diuretics": {"type": "numerical", "min": 1.0, "max": 5.0, "default": 3.0},
    "cs": {"type": "categorical", "options": [0, 1]},
    "rr": {"type": "numerical", "min": 0.0, "max": 60.0, "default": 18.0},
}

# Streamlit 界面
st.title("Prediction Model with SHAP Visualization")

# 动态生成输入项
st.header("Enter the following feature values:")
feature_values = []
for feature, properties in feature_ranges.items():
    if properties["type"] == "numerical":
        value = st.number_input(
            label=f"{feature} ({properties['min']} - {properties['max']})",
            min_value=float(properties["min"]),
            max_value=float(properties["max"]),
            value=float(properties["default"]),
        )
    elif properties["type"] == "categorical":
        value = st.selectbox(
            label=f"{feature} (Select a value)",
            options=properties["options"],
        )
    feature_values.append(value)

# 转换为模型输入格式
features = np.array([feature_values])

# 预测与 SHAP 可视化
if st.button("Predict"):
    # 模型预测
    predicted_class = model.predict(features)[1]
    predicted_proba = model.predict_proba(features)[1]

    # 提取预测的类别概率
    probability = predicted_proba[predicted_class] * 100

    # 显示预测结果，使用 Matplotlib 渲染指定字体
    text = f"Based on feature values, predicted possibility of AKI is {probability:.2f}%"
    fig, ax = plt.subplots(figsize=(8, 1))
    ax.text(
        0.5, 0.5, text,
        fontsize=16,
        ha='center', va='center',
        fontname='Times New Roman',
        transform=ax.transAxes
    )
    ax.axis('off')
    plt.savefig("prediction_text.png", bbox_inches='tight', dpi=300)
    st.image("prediction_text.png")

    # 计算 SHAP 值
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_ranges.keys()))

    # 生成 SHAP 力图
    class_index = predicted_class  # 当前预测类别
    shap_fig = shap.force_plot(
        explainer.expected_value[class_index],
        shap_values[:,:,class_index],
        pd.DataFrame([feature_values], columns=feature_ranges.keys()),
        matplotlib=True,
    )
    # 保存并显示 SHAP 图
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
