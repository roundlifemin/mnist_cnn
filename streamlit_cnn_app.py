import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import os
from scipy.ndimage import center_of_mass, shift
import matplotlib.pyplot as plt

# ----------------------------
# 모델 로딩
# ----------------------------
def get_latest_model():
    MODEL_DIR = "saved_models"
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".keras")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

model_path = get_latest_model()
model = tf.keras.models.load_model(model_path) if model_path else None

# ----------------------------
# 전처리 보조 함수
# ----------------------------
def enhance_contrast(image_arr):
    norm = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    blurred = cv2.GaussianBlur(norm, (5, 5), 0)
    inverted = cv2.bitwise_not(blurred)
    return inverted

def preprocess_and_predict(image_arr):
    results = {}

    # 대비 향상된 이미지
    enhanced_img = enhance_contrast(image_arr)

    # 다양한 이진화 방식
    methods = {
        "Adaptive Gaussian": cv2.adaptiveThreshold(enhanced_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2),
        "Otsu": cv2.threshold(enhanced_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        "Manual 100": np.where(enhanced_img > 100, 0, 255).astype("uint8")
    }

    for method_name, binary_img in methods.items():
        # (1) 선 굵기 강조: dilation
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(binary_img, kernel, iterations=1)

        # (2) 중심 정렬
        cy, cx = center_of_mass(dilated)
        shift_y = int(round(dilated.shape[0] // 2 - cy))
        shift_x = int(round(dilated.shape[1] // 2 - cx))
        shifted = shift(dilated, shift=(shift_y, shift_x), mode='constant', cval=0)

        # (3) 크기 변경
        resized = cv2.resize(shifted, (28, 28), interpolation=cv2.INTER_AREA)

        # (4) 정규화 + reshape
        norm = resized.astype("float32") / 255.0
        norm = np.clip(norm, 0.01, 1.0)  # 너무 작은 값 제거
        reshaped = norm.reshape(1, 28, 28, 1)

        # (5) 예측
        pred = model.predict(reshaped, verbose=0)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results[method_name] = {
            "processed": resized,
            "prediction": pred_class,
            "confidence": confidence,
            "prob": pred[0]
        }

    return results

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="웹캠 숫자 인식기", layout="centered")
st.title("📷 웹캠 숫자 인식기 (MNIST 기반 최종 개선 버전)")
st.markdown("흰 종이에 **굵은 검정색 펜**으로 숫자를 쓰고 웹캠으로 촬영해주세요.")

image_data = st.camera_input("📸 숫자를 찍어서 업로드해주세요:")

if image_data and model:
    image = Image.open(image_data).convert("L")
    gray_np = np.array(image)

    # 전처리 + 예측
    results = preprocess_and_predict(gray_np)
    best = max(results.items(), key=lambda x: x[1]['confidence'])
    best_label = best[1]['prediction']
    best_conf = best[1]['confidence']

    st.subheader(f"✅ 최종 예측: **{best_label}** (신뢰도: {best_conf:.2f})")
    st.bar_chart(best[1]['prob'])

    # 전처리별 비교
    st.subheader("🧪 전처리별 예측 결과")
    for method, data in results.items():
        st.markdown(f"**{method}** - 예측: {data['prediction']}, 신뢰도: {data['confidence']:.2f}")
        st.image(data['processed'], width=120)

    # 입력 이미지 히트맵
    st.subheader("🔥 입력 이미지 히트맵")
    fig, ax = plt.subplots()
    ax.imshow(gray_np, cmap='hot')
    ax.axis("off")
    st.pyplot(fig)

elif not model:
    st.warning("⚠️ 모델(.keras)이 없습니다. 먼저 학습한 모델을 `saved_models/` 폴더에 저장하세요.")
else:
    st.info("📸 웹캠으로 숫자 이미지를 먼저 촬영해주세요.")
