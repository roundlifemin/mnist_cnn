# streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt

# ---------------------------
# 최신 모델 경로 검색
# ---------------------------
MODEL_DIR = "saved_models"

def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

# ---------------------------
# 학습 로그 로딩
# ---------------------------
def load_training_log(log_path="saved_models/training_log.json"):
    if not os.path.exists(log_path):
        return None
    with open(log_path, "r") as f:
        return json.load(f)

def plot_training_log(log_data):
    st.subheader("학습 로그 (Accuracy / Loss)")
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
# 모델 로드
# ---------------------------
latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CNN 숫자 예측기 (MNIST)")
st.markdown("그림판에 **0~9** 숫자를 그리세요.")

# ---------------------------
# 학습 로그 출력
# ---------------------------
log_data = load_training_log()
if log_data:
    plot_training_log(log_data)

# ---------------------------
# 캔버스
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=30,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------------------
# 예측 실행
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data[:, :, 0]
    img = Image.fromarray(img.astype("uint8")).convert("L")  # 흑백화
    img = ImageOps.invert(img)

    # 이진화
    img_arr = np.array(img)
    img_arr = (img_arr > 100).astype("uint8") * 255

    # 중심 이동
    cy, cx = center_of_mass(img_arr)
    shift_y = img_arr.shape[0] // 2 - cy
    shift_x = img_arr.shape[1] // 2 - cx
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 리사이즈 및 정규화
    img = Image.fromarray(img_arr.astype("uint8"))
    img = img.resize((28, 28))
    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    # 예측
    pred = model.predict(img_arr)
    pred_class = int(np.argmax(pred))

    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])

elif not model:
    st.warning("모델이 없습니다. 먼저 학습해 주세요.")
