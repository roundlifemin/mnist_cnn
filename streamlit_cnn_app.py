# streamlit_cnn_debug_app.py
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import os

# ---------------------------
# 모델 불러오기
# ---------------------------
MODEL_PATH = "saved_models"
model_files = sorted([f for f in os.listdir(MODEL_PATH) if f.endswith(".keras")], reverse=True)
model_path = os.path.join(MODEL_PATH, model_files[0]) if model_files else None
model = tf.keras.models.load_model(model_path) if model_path else None

# ---------------------------
# UI
# ---------------------------
st.title("🧠 CNN 숫자 예측기 (디버그 모드)")
st.markdown("그림판에 숫자를 그려서 예측 결과 및 내부 전처리 과정을 확인합니다.")

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
    key="canvas",
)

# ---------------------------
# 이미지 전처리 함수
# ---------------------------
def preprocess_canvas(canvas_img):
    img = canvas_img[:, :, 0]  # grayscale
    img = Image.fromarray(img.astype("uint8")).convert("L")
    img = ImageOps.invert(img)

    # 이진화
    arr = np.array(img)
    arr[arr < 100] = 0
    arr[arr >= 100] = 255

    # 중심 이동
    cy, cx = center_of_mass(arr)
    shift_y = int(arr.shape[0] // 2 - cy)
    shift_x = int(arr.shape[1] // 2 - cx)
    arr = shift(arr, shift=(shift_y, shift_x), mode='constant', cval=0)

    # 크기 조정 및 정규화
    img = Image.fromarray(arr.astype("uint8")).resize((28, 28))
    norm = np.array(img).astype("float32") / 255.0
    norm = norm.reshape(1, 28, 28, 1)

    return img, norm

# ---------------------------
# 예측 실행
# ---------------------------
if st.button("예측 실행") and canvas_result.image_data is not None:
    if model is None:
        st.error("모델이 로드되지 않았습니다.")
    else:
        processed_img, input_tensor = preprocess_canvas(canvas_result.image_data)

        # 예측
        prediction = model.predict(input_tensor)
        predicted_label = int(np.argmax(prediction))

        st.subheader(f"🎯 예측된 숫자: **{predicted_label}**")
        st.bar_chart(prediction[0])

        # 이미지 디버깅 저장
        os.makedirs("debug_output", exist_ok=True)
        processed_img.save("debug_output/canvas_input.png")

        # 시각화
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(canvas_result.image_data, caption="🖍 원본 캔버스 (280x280)", use_column_width=True)
        with col2:
            st.image(processed_img, caption="🎨 전처리 후 이미지 (28x28)", use_column_width=True)
        with col3:
            fig, ax = plt.subplots()
            ax.imshow(np.array(processed_img), cmap="hot")
            ax.set_title("🔥 입력 히트맵")
            ax.axis("off")
            st.pyplot(fig)

        st.success("예측 및 디버깅 이미지 저장 완료: `debug_output/canvas_input.png`")

elif not model:
    st.warning("모델을 먼저 학습해 주세요.")
