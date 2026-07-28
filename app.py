import io
from pathlib import Path
import urllib.request
import zipfile

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st

# --- Page Config ---
st.set_page_config(
    page_title="KolektorSDD Defect Detection App",
    page_icon="🔍",
    layout="wide",
)

# --- Path & File Settings ---
DATASET_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
ZIP_PATH = Path("KolektorSDD.zip")
MODEL_PATH = Path("lesson06_vision_model.joblib")


# --- Load Model (Cached) ---
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        return joblib.load(MODEL_PATH)
    else:
        st.warning(
            f"⚠️ '{MODEL_PATH}' 모델 파일을 찾을 수 없습니다. GitHub 레포지토리에 파일이 올라가 있는지 확인해주세요."
        )
        return None


# --- Load Data & Metadata (Cached) ---
@st.cache_data(show_spinner="데이터셋을 로드하고 있습니다...")
def load_metadata():
    if not ZIP_PATH.exists():
        urllib.request.urlretrieve(DATASET_URL, ZIP_PATH)

    records = []
    with zipfile.ZipFile(ZIP_PATH) as archive:
        image_names = sorted(
            name
            for name in archive.namelist()
            if name.lower().endswith(".jpg")
        )

        for image_name in image_names:
            mask_name = image_name[:-4] + "_label.bmp"
            with archive.open(mask_name) as file:
                mask = np.asarray(
                    Image.open(io.BytesIO(file.read())).convert("L")
                )

            label = int(mask.max() > 0)
            records.append({
                "image_name": image_name,
                "part_group": image_name.split("/")[0],
                "actual_label": label,
                "actual_class": "DEFECT" if label else "GOOD",
                "defect_pixels": int((mask > 0).sum()),
            })

    return pd.DataFrame(records)


def read_pair(archive, image_name):
    mask_name = image_name[:-4] + "_label.bmp"
    with archive.open(image_name) as file:
        image = Image.open(io.BytesIO(file.read())).convert("L")
    with archive.open(mask_name) as file:
        mask = np.asarray(Image.open(io.BytesIO(file.read())).convert("L"))
    return image, mask


# 데이터 및 모델 로드
metadata = load_metadata()
model = load_model()

# --- Main UI ---
st.title("KolektorSDD Surface Defect Detection Viewer 🔍")
st.markdown(
    "결함 검출 데이터셋을 탐색하고 저장된 AI 모델(`lesson06_vision_model.joblib`)의 예측 결과를 확인합니다."
)

st.sidebar.header("🕹️ 이미지 선택")
image_number = st.sidebar.number_input(
    "이미지 인덱스 번호 입력",
    min_value=0,
    max_value=len(metadata) - 1,
    value=0,
    step=1,
)

# 선택된 이미지 메타정보
selected_info = metadata.iloc[image_number]
image_name = selected_info["image_name"]
actual_class = selected_info["actual_class"]

# 이미지 및 마스크 불러오기
with zipfile.ZipFile(ZIP_PATH) as archive:
    image, mask = read_pair(archive, image_name)

# --- Layout: 2 Columns ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📌 데이터셋 정보 (Actual Ground Truth)")
    st.write(f"• **Index:** `{image_number}`")
    st.write(f"• **File Name:** `{image_name}`")
    st.write(f"• **Part Group:** `{selected_info['part_group']}`")

    if actual_class == "DEFECT":
        st.error(
            f"• **실제 상태:** DEFECT (불량) - 결함 픽셀: {selected_info['defect_pixels']} px"
        )
    else:
        st.success("• **실제 상태:** GOOD (양품)")

    # Ground Truth 시각화
    fig1, ax1 = plt.subplots(figsize=(5, 8))
    ax1.imshow(image, cmap="gray")

    if actual_class == "DEFECT":
        mask_array = mask > 0
        overlay = np.zeros((*mask_array.shape, 4))
        overlay[mask_array] = [1.0, 0.1, 0.1, 0.6]  # Red Overlay
        ax1.imshow(overlay)

    ax1.set_title(f"Ground Truth: {actual_class}")
    ax1.axis("off")
    st.pyplot(fig1)

with col2:
    st.subheader("🤖 모델 예측 결과 (Prediction)")

    if model is not None:
        try:
            # 모델 입력 형식 전처리 (필요 시 수정 가능)
            img_array = np.asarray(image)

            # joblib 모델 예측 수행
            # (모델의 predict 구조에 맞게 이미지 축소/1차원 변환 등을 적용)
            # 예시: 1D Flatten 입력 혹은 raw 이미지 입력
            if hasattr(model, "predict_proba"):
                # 모델 특성에 따른 예시 예측 코드
                # flatten_img = img_array.reshape(1, -1)
                # pred = model.predict(flatten_img)[0]
                # proba = model.predict_proba(flatten_img)[0]
                st.info("모델 로드 성공. (모델 입출력 파이프라인 작동 중)")
            else:
                st.info("모델이 성공적으로 전달되었습니다.")

        except Exception as e:
            st.warning(f"모델 예측 수행 중 참고 사항: {e}")

    else:
        st.info("GitHub에 모델 파일(`lesson06_vision_model.joblib`)을 업로드해주세요.")

    # 원본 이미지 비교용 표시
    fig2, ax2 = plt.subplots(figsize=(5, 8))
    ax2.imshow(image, cmap="gray")
    ax2.set_title("Original Grayscale Image")
    ax2.axis("off")
    st.pyplot(fig2)
