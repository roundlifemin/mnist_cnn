# streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import cv2

# ---------------------------
# 모델 로드
# ---------------------------
MODEL_DIR = "saved_models"
def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# 앱 UI
# ---------------------------
st.title("CNN 기반 숫자 예측기 (MNIST)")
st.markdown("그림판에 **0~9** 숫자를 직접 그려보세요.")

# ---------------------------
# 그리기 캔버스
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------------------
# 예측 버튼
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data[:, :, 0]  # 흑백 채널만
    img = Image.fromarray(img.astype("uint8")).resize((28, 28))
    img = ImageOps.invert(img)

    # NumPy 변환 및 이진화 (threshold ↓)
    img_arr = np.array(img)
    img_arr = cv2.GaussianBlur(img_arr, (3, 3), 0)  # 흐림 효과 추가
    img_arr[img_arr < 80] = 0
    img_arr[img_arr >= 80] = 255

    # Padding 추가 (MNIST 대비 캔버스는 외곽으로 붙는 경향 있음)
    img_arr = np.pad(img_arr, pad_width=4, mode='constant', constant_values=0)
    img_arr = cv2.resize(img_arr, (28, 28))  # 다시 28x28로 축소

    # 중심 정렬
    cy, cx = center_of_mass(img_arr)
    shift_y = int(14 - cy)
    shift_x = int(14 - cx)
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 정규화 및 reshape
    img_arr = img_arr.astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    # 예측
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred)

    # 결과 출력
    st.subheader(f"🎯 예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])

    # 디버깅용 시각화
    st.markdown("**전처리된 입력 이미지 (패딩, 블러, 중심정렬)**")
    fig, ax = plt.subplots()
    ax.imshow(img_arr[0].reshape(28, 28), cmap='gray')
    ax.axis("off")
    st.pyplot(fig)

elif not model:
    st.warning("모델이 없습니다. 먼저 학습해 주세요.")
