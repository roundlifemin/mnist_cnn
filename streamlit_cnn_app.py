import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps
import cv2
import io
from scipy.ndimage import center_of_mass, shift
import os
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
# 타이틀
# ----------------------------
st.title("웹캠 숫자 인식기 (MNIST 기반 개선 버전)")
st.markdown("흰 종이에 검은색 펜으로 0~9 숫자를 작성 후 웹캠으로 촬영해보세요.")

# ----------------------------
# 웹캠 입력
# ----------------------------
image_data = st.camera_input("숫자가 보이도록 웹캠으로 촬영")

def apply_preprocessing(image_arr):
    results = {}

    # 히스토그램 평활화 (명암 대비 향상)
    norm_img = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")

    # 여러 전처리 방법 적용
    methods = {
        "Adaptive Gaussian": cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2),
        "Otsu": cv2.threshold(norm_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        "Manual 100": np.where(norm_img > 100, 0, 255).astype("uint8")
    }

    for key, img in methods.items():
        # 중심 이동
        cy, cx = center_of_mass(img)
        shift_y = int(round(img.shape[0] // 2 - cy))
        shift_x = int(round(img.shape[1] // 2 - cx))
        shifted = shift(img, shift=(shift_y, shift_x), mode='constant', cval=0)

        # 리사이즈 및 정규화
        resized = cv2.resize(shifted, (28, 28), interpolation=cv2.INTER_AREA)
        norm = resized.astype("float32") / 255.0
        reshaped = norm.reshape(1, 28, 28, 1)

        # 예측
        pred = model.predict(reshaped, verbose=0)
        pred_class = int(np.argmax(pred))
        confidence = float(np.max(pred))

        results[key] = {
            "processed": resized,
            "prediction": pred_class,
            "confidence": confidence,
            "prob": pred[0]
        }

    return results

# ----------------------------
# 예측 처리
# ----------------------------
if image_data is not None and model:
    # 이미지 로드
    image = Image.open(image_data).convert("L")  # 그레이스케일
    gray_np = np.array(image)

    # 전처리 적용
    results = apply_preprocessing(gray_np)

    # 결과 선택
    best = max(results.items(), key=lambda x: x[1]['confidence'])
    best_label = best[1]['prediction']
    best_conf = best[1]['confidence']

    st.subheader(f"최종 예측: **{best_label}** (신뢰도: {best_conf:.2f})")
    st.bar_chart(best[1]['prob'])

    # 전처리별 결과 출력
    for method, data in results.items():
        st.markdown(f"### {method} (예측: {data['prediction']}, 신뢰도: {data['confidence']:.2f})")
        st.image(data['processed'], width=140)

    # 입력 히트맵 출력
    st.subheader("입력 이미지 히트맵")
    fig, ax = plt.subplots()
    ax.imshow(gray_np, cmap='hot')
    ax.axis("off")
    st.pyplot(fig)

else:
    st.info("먼저 웹캠으로 숫자를 촬영해주세요.")
