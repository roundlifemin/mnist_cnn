# streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import matplotlib.pyplot as plt
import os

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
    img = canvas_result.image_data[:, :, 0]  # 흑백 채널만 사용
    img = Image.fromarray(img.astype("uint8")).resize((28, 28))  # 크기 조정

    # 색 반전 (MNIST: 흰 배경에 검정 숫자)
    img = ImageOps.invert(img)

    # NumPy 변환 및 이진화
    img_arr = np.array(img)
    img_arr[img_arr < 128] = 0
    img_arr[img_arr >= 128] = 255

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

    # 출력
    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])

    # 디버깅: 전처리 이미지 확인
    st.markdown("**전처리된 입력 이미지** (중심정렬, 이진화 적용):")
    fig, ax = plt.subplots()
    ax.imshow(img_arr[0].reshape(28, 28), cmap='gray')
    ax.axis("off")
    st.pyplot(fig)

elif not model:
    st.warning("모델이 없습니다. 먼저 학습해 주세요.")
