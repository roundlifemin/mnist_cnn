# streamlit_cnn_app.py
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
import json
from tensorflow.keras.datasets import mnist
from PIL import Image

# ---------------------------
# 모델 및 로그 로딩 함수
# ---------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

def load_training_log(log_path="saved_models/training_log.json"):
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return None

def plot_training_log(log_data):
    st.subheader("📈 학습 로그 (Accuracy / Loss)")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    ax[0].plot(log_data["accuracy"], label="Train Acc")
    ax[0].plot(log_data["val_accuracy"], label="Val Acc")
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(log_data["loss"], label="Train Loss")
    ax[1].plot(log_data["val_loss"], label="Val Loss")
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)

# ---------------------------
# 데이터 로딩 (X_test)
# ---------------------------
(_, _), (X_test, y_test) = mnist.load_data()
X_test = X_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ---------------------------
# 모델 로드
# ---------------------------
latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="MNIST Test Sample Prediction", layout="centered")
st.title("CNN 숫자 예측기 (MNIST 샘플 선택)")
st.markdown("`X_test`의 실제 손글씨 샘플을 선택하여 CNN 모델이 예측합니다.")

# ---------------------------
# 학습 로그 시각화
# ---------------------------
log_data = load_training_log()
if log_data:
    plot_training_log(log_data)
else:
    st.info(" 학습 로그 파일이 없거나 비어 있습니다.")

# ---------------------------
# 테스트 샘플 선택
# ---------------------------
if model:
    st.markdown("### 테스트 샘플 선택")
    sample_index = st.slider("샘플 인덱스 선택 (0~9999)", min_value=0, max_value=9999, value=0)

    img = X_test[sample_index].reshape(28, 28)
    label = y_test[sample_index]

    st.image(img, caption=f"실제 숫자: {label}", width=150)

    # 예측
    pred = model.predict(X_test[sample_index].reshape(1, 28, 28, 1), verbose=0)
    pred_class = int(np.argmax(pred))

    st.subheader(f" 예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])
else:
    st.warning("모델이 없습니다. 먼저 학습을 완료하고 다시 실행해주세요.")
