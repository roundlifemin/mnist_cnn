import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import os

# ---------------------------
# 최신 모델 로드 함수
# ---------------------------
MODEL_DIR = "saved_models"
def get_latest_model():
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path, compile=False) if latest_model_path else None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("CNN 기반 숫자 예측기 (MNIST)")
st.markdown("그림판에 **0~9** 숫자를 그려보세요.")

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
# 예측 처리
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None and model:
    img = canvas_result.image_data[:, :, 0]
    img = Image.fromarray(img.astype("uint8")).resize((28, 28))
    img = ImageOps.invert(img)

    # 이진화
    img_arr = np.array(img)
    img_arr[img_arr < 100] = 0
    img_arr[img_arr >= 100] = 255

    # 중심 이동
    cy, cx = center_of_mass(img_arr)
    shift_y = int(14 - cy)
    shift_x = int(14 - cx)
    img_arr = shift(img_arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 정규화 및 CNN용 reshape
    img_arr = img_arr.astype("float32") / 255.0
    img_arr = img_arr.reshape(1, 28, 28, 1)

    pred = model.predict(img_arr)
    pred_class = np.argmax(pred)

    st.subheader(f"예측된 숫자: **{pred_class}**")
    st.bar_chart(pred[0])
elif not model:
    st.warning("모델이 없습니다. 먼저 학습해 주세요.")
