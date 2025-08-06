%%writefile streamlit_cnn_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf
import os
from scipy.ndimage import center_of_mass, shift

# ---------------------------
# 모델 로드 함수
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
# Streamlit UI 구성
# ---------------------------
st.title("🧠 CNN 기반 MNIST 숫자 예측기")
st.markdown("그림판에 숫자(0~9)를 **굵고 명확하게 중앙에** 그려주세요.")

# ---------------------------
# 그리기 캔버스
# ---------------------------
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=20,                # 선 굵기 증가
    stroke_color="#FFFFFF",
    background_color="#000000",
    width=280,
    height=280,
    drawing_mode="freedraw",
    key="canvas"
)

# ---------------------------
# 예측 처리
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    # 1. 흑백 채널만 사용
    img = canvas_result.image_data[:, :, 0]
    img = Image.fromarray(img.astype("uint8"))

    # 2. 28x28 크기로 축소 + 색 반전
    img = img.resize((28, 28))
    img = ImageOps.invert(img)

    # 3. NumPy로 변환 후 이진화 (흐린 선 제거)
    img_arr = np.array(img)
    img_arr[img_arr < 100] = 0
    img_arr[img_arr >= 100] = 255

    # 4. 중심 정렬 (무게중심 이동)
    cy, cx = center_of_mass(img_arr)
    shift_y = int(img_arr.shape[0] / 2 - cy)
    shift_x = int(img_arr.shape[1] / 2 - cx)
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 5. 정규화 및 reshape (CNN 입력형식)
    img_arr = img_arr.astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    # 6. 예측
    pred = model.predict(img_arr)
    pred_class = np.argmax(pred)

    # 7. 결과 출력
    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])
    st.image(img_arr.reshape(28, 28), width=150, caption="전처리된 입력", clamp=True)

elif not model:
    st.warning("모델이 없습니다. 먼저 CNN 모델을 학습하고 `.h5` 형식으로 저장해 주세요.")
