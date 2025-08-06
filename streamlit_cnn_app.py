import streamlit as st
from streamlit_drawable_canvas import st_canvas
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import center_of_mass, shift
import tensorflow as tf
import os
import json
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

st.set_page_config(layout="wide")

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


# ---------------------------
# adaptive thresholding
# ---------------------------
def adaptive_binarize(img):
    arr = np.array(img)
    mean = np.mean(arr)
    binarized = (arr > mean * 0.7).astype("uint8") * 255
    return binarized


# ---------------------------
# 중심 이동 보완 함수
# ---------------------------
def better_center_shift(arr):
    cy, cx = center_of_mass(arr)
    shift_y = int(arr.shape[0] / 2 - cy)
    shift_x = int(arr.shape[1] / 2 - cx)
    shift_y = np.clip(shift_y, -10, 10)
    shift_x = np.clip(shift_x, -10, 10)
    shifted = shift(arr, shift=(shift_y, shift_x), mode='constant', cval=0)
    return shifted, (cy, cx)


# ---------------------------
# 모델 로드
# ---------------------------
latest_model_path = get_latest_model()
model = tf.keras.models.load_model(latest_model_path) if latest_model_path else None

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🧠 CNN 숫자 예측기 (디버그 모드)")
st.markdown("그림판에 숫자를 그려서 예측 결과 및 내부 전처리 과정을 확인합니다.")

# ---------------------------
# 학습 로그 출력
# ---------------------------
log_data = load_training_log()
if log_data:
    st.subheader("학습 로그")
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(log_data["accuracy"], label="Train Acc")
    ax[0].plot(log_data["val_accuracy"], label="Val Acc")
    ax[0].set_title("Accuracy")
    ax[0].legend()
    ax[1].plot(log_data["loss"], label="Train Loss")
    ax[1].plot(log_data["val_loss"], label="Val Loss")
    ax[1].set_title("Loss")
    ax[1].legend()
    st.pyplot(fig)

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
    img = Image.fromarray(img.astype("uint8")).convert("L")
    img = ImageOps.invert(img)

    # adaptive threshold
    bin_img = adaptive_binarize(img)

    # 중심 이동
    shifted_img, center = better_center_shift(bin_img)

    # 28x28 resize
    img_pil = Image.fromarray(shifted_img.astype("uint8"))
    img_pil = img_pil.resize((28, 28))
    norm_arr = np.array(img_pil).astype("float32") / 255.0
    input_tensor = norm_arr.reshape(1, 28, 28, 1)

    # 예측
    pred = model.predict(input_tensor)
    pred_class = int(np.argmax(pred))

    st.subheader(f"예측된 숫자: {pred_class}")
    st.bar_chart(pred[0])

    # 디버그용 시각화
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(canvas_result.image_data, caption="원본 입력")
    with col2:
        st.image(norm_arr, caption="28x28 정규화")
    with col3:
        fig, ax = plt.subplots()
        heatmap = ax.imshow(norm_arr, cmap='hot')
        ax.set_title(f"히트맵 (Center: {center[0]:.1f}, {center[1]:.1f})")
        ax.axis("off")
        st.pyplot(fig)

    # 원래 MNIST 이미지 비교
    st.subheader("🆚 MNIST Test 샘플 비교")
    (_, _), (X_test, y_test) = mnist.load_data()
    test_idx = st.slider("MNIST 샘플 번호", min_value=0, max_value=9999, value=0)
    st.image(X_test[test_idx], caption=f"정답: {y_test[test_idx]}")

elif not model:
    st.warning("모델이 없습니다. 먼저 학습을 진행하세요.")
