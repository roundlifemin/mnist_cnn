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
        "Adaptive Gaussian": cv2.adaptiveThreshold(norm_img, 255, cv2.ADAPTIVE_THRESH
