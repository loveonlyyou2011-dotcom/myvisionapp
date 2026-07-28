import io
import os
from pathlib import Path
from typing import Any
import urllib.request
import zipfile

joblib = None
try:
    import joblib
except ImportError:
    pass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError
import streamlit as st

# --- 상수 및 경로 설정 ---
MODEL_FILENAME = "lesson06_vision_model.joblib"
DATASET_URL = "https://data.vicos.si/datasets/KSDD/KolektorSDD.zip"
ZIP_PATH = Path("KolektorSDD.zip")
MAX_UPLOAD_BYTES = 10 * 1024 * 1024

st.set_page_config(
    page_title="6차시 비전검사 모델 체험 및 KolektorSDD 뷰어",
    page_icon="🔍",
    layout="wide",
)


# --- 1. 모델 경로 탐색 및 로딩 ---
def model_candidates() -> list[Path]:
    candidates: list[Path] = []
    configured = os.getenv("LESSON06_MODEL_PATH")
    if configured:
        candidates.append(Path(configured).expanduser())

    candidates.extend(
        [
            Path.cwd() / MODEL_FILENAME,
            Path.cwd() / "lesson06_outputs" / MODEL_FILENAME,
            Path("/content/lesson06_outputs") / MODEL_FILENAME,
            Path("/content") / MODEL_FILENAME,
        ]
    )

    script_path = Path(__file__).resolve()
    if len(script_path.parents) >= 3:
        textbook_root = script_path.parents[2]
        candidates.append(textbook_root / "outputs" / "day6" / MODEL_FILENAME)

    return list(dict.fromkeys(path.resolve() for path in candidates))


def find_model_path() -> Path:
    for path in model_candidates():
        if path.is_file():
            return path
    searched = "\n".join(f"- {path}" for path in model_candidates())
    raise FileNotFoundError(
        f"{MODEL_FILENAME}을 찾지 못했습니다.\n검색 위치:\n{searched}"
    )


@st.cache_resource
def load_bundle(model_path: str) -> dict[str, Any]:
    if joblib is None:
        raise ImportError(
            "joblib 패키지가 필요합니다. requirements.txt를 확인해주세요."
        )
    bundle = joblib.load(model_path)
    required = {
        "model",
        "feature_size",
        "operating_threshold",
        "quality_limits",
        "class_names",
        "feature_extractor",
    }
    missing = required.difference(bundle)
    if missing:
        raise ValueError(
            "모델 번들에 필요한 항목이 없습니다: "
            + ", ".join(sorted(missing))
        )
    if bundle["feature_extractor"] != "lesson06_hog_intensity_v1":
        raise ValueError(
            "이 앱과 호환되지 않는 특징 추출기입니다: "
            f"{bundle['feature_extractor']}"
        )
    return bundle


# --- 2. 데이터셋 로딩 (캐싱) ---
@st.cache_data(show_spinner="KolektorSDD 데이터셋을 준비하는 중입니다...")
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


# --- 3. 이미지 품질 계산 및 HOG 특징 추출 ---
def quality_metrics(
    image: Image.Image, feature_size: tuple[int, int]
) -> dict[str, float]:
    array = np.asarray(image.resize(feature_size), dtype=np.float32)
    gx = np.diff(array, axis=1, prepend=array[:, :1])
    gy = np.diff(array, axis=0, prepend=array[:1, :])
    laplacian = (
        -4 * array
        + np.roll(array, 1, axis=0)
        + np.roll(array, -1, axis=0)
        + np.roll(array, 1, axis=1)
        + np.roll(array, -1, axis=1)
    )
    return {
        "brightness": float(array.mean()),
        "contrast": float(array.std()),
        "sharpness": float(laplacian.var()),
        "mean_gradient": float(np.hypot(gx, gy).mean()),
    }


def extract_features(
    image: Image.Image, feature_size: tuple[int, int]
) -> np.ndarray:
    array = np.asarray(
        image.resize(feature_size, Image.Resampling.BILINEAR), dtype=np.float32
    )
    normalized = array / 255.0
    gx = np.diff(normalized, axis=1, prepend=normalized[:, :1])
    gy = np.diff(normalized, axis=0, prepend=normalized[:1, :])
    magnitude = np.hypot(gx, gy)
    orientation = (np.degrees(np.arctan2(gy, gx)) + 180) % 180

    hog: list[float] = []
    bins = np.linspace(0, 180, 10)
    for row in range(0, feature_size[1], 8):
        for column in range(0, feature_size[0], 8):
            cell_angle = orientation[row : row + 8, column : column + 8]
            cell_weight = magnitude[row : row + 8, column : column + 8]
            histogram, _ = np.histogram(
                cell_angle, bins=bins, weights=cell_weight
            )
            histogram = histogram / (histogram.sum() + 1e-6)
            hog.extend(histogram.tolist())

    intensity_histogram, _ = np.histogram(
        normalized, bins=16, range=(0, 1), density=True
    )
    percentiles = np.percentile(normalized, [1, 5, 25, 50, 75, 95, 99])
    extra = np.array([
        normalized.mean(),
        normalized.std(),
        magnitude.mean(),
        np.percentile(magnitude, 90),
        np.percentile(magnitude, 99),
    ])
    return np.concatenate(
        [np.asarray(hog), intensity_histogram, percentiles, extra]
    )


def quality_failures(
    metrics: dict[str, float], limits: dict[str, float]
) -> list[str]:
    failures = []
    if metrics["brightness"] < limits["brightness_low"]:
        failures.append("밝기가 학습 범위보다 낮습니다.")
    if metrics["brightness"] > limits["brightness_high"]:
        failures.append("밝기가 학습 범위보다 높습니다.")
    if metrics["contrast"] < limits["contrast_low"]:
        failures.append("대비가 부족합니다.")
    if metrics["sharpness"] < limits["sharpness_low"]:
        failures.append("초점 또는 해상도가 부족합니다.")
    return failures


# --- 4. 공통 예측 추론 UI 함수 ---
def run_model_inference(image: Image.Image, bundle: dict[str, Any]):
    feature_size = tuple(int(value) for value in bundle["feature_size"])
    metrics = quality_metrics(image, feature_size)
    failures = quality_failures(metrics, bundle["quality_limits"])

    st.subheader("1. 이미지 품질 게이트")
    m_cols = st.columns(3)
    m_cols[0].metric("밝기", f"{metrics['brightness']:.1f}")
    m_cols[1].metric("대비", f"{metrics['contrast']:.1f}")
    m_cols[2].metric("선명도", f"{metrics['sharpness']:.1f}")

    if failures:
        st.error("❌ 촬영 조건 부적합 — 재촬영 또는 사람 검토 필요")
        for failure in failures:
            st.write(f"- {failure}")
        st.info(
            "품질 게이트를 통과하지 못해 모델 예측 점수를 계산하지 않습니다."
        )
        return

    st.success("✅ 촬영 품질 기준 통과")
    st.subheader("2. AI 모델 예측 결과")

    feature = extract_features(image, feature_size).reshape(1, -1)
    expected_features = getattr(
        bundle["model"], "n_features_in_", feature.shape[1]
    )

    if feature.shape[1] != expected_features:
        st.error(
            f"특징 수가 저장 모델과 일치하지 않습니다. (현재: {feature.shape[1]}개, 모델 요구: {expected_features}개)"
        )
        return

    defect_probability = float(bundle["model"].predict_proba(feature)[0, 1])
    threshold = float(bundle["operating_threshold"])
    is_review_candidate = defect_probability >= threshold

    st.metric("모델 불량 점수", f"{defect_probability * 100:.1f}%")
    st.progress(min(max(defect_probability, 0.0), 1.0))
    st.caption(f"운영 검토 임계값: {threshold * 100:.1f}%")

    if is_review_candidate:
        st.error("🚨 판정: **불량 검토 후보 (DEFECT)**")
    else:
        st.success("🟢 판정: **정상 후보 (GOOD)**")


# --- 5. Main App Render ---
def render_app() -> None:
    st.title("6차시 비전검사 모델 체험 및 KolektorSDD 뷰어 🔍")
    st.caption(
        "KolektorSDD 이미지로 학습한 교육용 기준선 모델입니다. 모델 출력은 불량 확정이 아닌 사람 검토 후보입니다."
    )

    # 모델 불러오기
    try:
        model_path = find_model_path()
        bundle = load_bundle(str(model_path))
    except Exception as error:
        st.error("저장 모델(`lesson06_vision_model.joblib`)을 불러오지 못했습니다.")
        st.code(str(error))
        st.info(
            f"노트북에서 학습을 완료하거나 `{MODEL_FILENAME}` 파일을 레포지토리에 올렸는지 확인해주세요."
        )
        st.stop()

    # 데이터셋 불러오기
    try:
        metadata = load_metadata()
    except Exception as e:
        st.warning(f"KolektorSDD 데이터셋을 다운로드하는 중 문제가 발생했습니다: {e}")
        metadata = None

    # 사이드바 정보
    with st.sidebar:
        st.subheader("🤖 모델 정보")
        st.write(f"• **파일 경로:** `{model_path.name}`")
        st.write(f"• **데이터셋:** {bundle.get('dataset', 'KolektorSDD')}")
        st.write(f"• **운영 임계값:** `{float(bundle['operating_threshold']):.2f}`")
        st.write(f"• **특징 추출기:** `{bundle['feature_extractor']}`")
        st.warning("다른 카메라·조명 환경의 이미지는 학습 범위 밖일 수 있습니다.")

    # 메인 탭 설정
    tab1, tab2 = st.tabs([
        "🖼️ 데이터셋 인덱스 조희",
        "📤 사용자 이미지 업로드",
    ])

    # --- TAB 1: 데이터셋 번호 탐색 ---
    with tab1:
        if metadata is not None:
            image_number = st.number_input(
                "조회할 이미지 인덱스 번호 선택",
                min_value=0,
                max_value=len(metadata) - 1,
                value=0,
                step=1,
            )

            selected_info = metadata.iloc[image_number]
            image_name = selected_info["image_name"]
            actual_class = selected_info["actual_class"]

            with zipfile.ZipFile(ZIP_PATH) as archive:
                image, mask = read_pair(archive, image_name)

            col1, col2 = st.columns([1, 1.2])

            with col1:
                st.subheader("Ground Truth (실제 데이터)")
                st.write(f"• **파일명:** `{image_name}`")
                st.write(f"• **그룹:** `{selected_info['part_group']}`")

                if actual_class == "DEFECT":
                    st.error(
                        f"• **실제 상태:** DEFECT (결함 픽셀: {selected_info['defect_pixels']}px)"
                    )
                else:
                    st.success("• **실제 상태:** GOOD (양품)")

                fig, ax = plt.subplots(figsize=(5, 7))
                ax.imshow(image, cmap="gray")
                if actual_class == "DEFECT":
                    mask_array = mask > 0
                    overlay = np.zeros((*mask_array.shape, 4))
                    overlay[mask_array] = [1.0, 0.1, 0.1, 0.6]  # Red overlay
                    ax.imshow(overlay)
                ax.set_title(f"Index {image_number}: {actual_class}")
                ax.axis("off")
                st.pyplot(fig)

            with col2:
                # 데이터셋 이미지에 대한 AI 모델 예측 진행
                run_model_inference(image, bundle)
        else:
            st.warning("데이터셋 메타데이터를 불러올 수 없습니다.")

    # --- TAB 2: 사용자 파일 업로드 ---
    with tab2:
        uploaded = st.file_uploader(
            "검사할 JPG 또는 PNG 이미지 한 장을 선택하세요.",
            type=["jpg", "jpeg", "png"],
            key="user_upload_key",
        )

        if uploaded is None:
            st.info("이미지를 선택하면 촬영 품질을 검사한 뒤 모델을 실행합니다.")
        elif uploaded.size > MAX_UPLOAD_BYTES:
            st.error("파일 크기는 10MB 이하여야 합니다.")
        else:
            try:
                up_image = Image.open(uploaded)
                up_image.verify()
                uploaded.seek(0)
                up_image = Image.open(uploaded).convert("L")

                up_col1, up_col2 = st.columns([1, 1.2])
                with up_col1:
                    st.subheader("업로드된 이미지")
                    st.image(
                        up_image, caption="검사 대상 이미지", use_container_width=True
                    )

                with up_col2:
                    run_model_inference(up_image, bundle)

            except (UnidentifiedImageError, OSError, ValueError):
                st.error("지원되는 정상적인 이미지 파일이 아닙니다.")

    st.markdown("---")
    st.warning(
        "⚠️ 이 결과만으로 제품을 폐기하거나 공정을 정지하지 마십시오. "
        "승인된 검사 기준에 따라 작업자가 원본 이미지와 제품을 최종 확인해야 합니다."
    )


if __name__ == "__main__":
    render_app()
