# ============================================================
# 직접 그린 숫자 예측 - PyTorch 버전
# ============================================================

import os

import streamlit as st
from streamlit_drawable_canvas import st_canvas

import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageOps

from scipy.ndimage import center_of_mass, shift

import cv2

import torch
import torch.nn as nn


# ------------------------------------------------------------
# 모델 설정
# ------------------------------------------------------------
MODEL_DIR = "saved_models"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ------------------------------------------------------------
# CNN 모델
# ------------------------------------------------------------
class MNISTCNN(nn.Module):

    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ------------------------------------------------------------
# 최신 모델 찾기
# ------------------------------------------------------------
def get_latest_model():

    if not os.path.exists(MODEL_DIR):
        return None

    models = [
        f for f in os.listdir(MODEL_DIR)
        if f.endswith(".pth")
    ]

    if not models:
        return None

    models.sort(reverse=True)

    return os.path.join(
        MODEL_DIR,
        models[0]
    )


# ------------------------------------------------------------
# 모델 로드
# ------------------------------------------------------------
model_path = get_latest_model()

model = None

if model_path:

    model = MNISTCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="MNIST 직접 입력",
    layout="centered"
)

st.title(
    "CNN 숫자 예측기 (MNIST) - PyTorch"
)

st.markdown(
    "검은 배경에 흰색으로 숫자를 그린 후 "
    "CNN 모델의 예측 결과를 확인합니다."
)


# ------------------------------------------------------------
# Canvas
# ------------------------------------------------------------
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


# ------------------------------------------------------------
# 이미지 전처리
# ------------------------------------------------------------
@st.cache_data
def apply_preprocessing(image_arr):

    results = {}

    # 0~255 범위로 정규화
    norm_img = cv2.normalize(
        image_arr,
        None,
        0,
        255,
        cv2.NORM_MINMAX
    ).astype("uint8")

    methods = {

        "Adaptive Gaussian":
            cv2.adaptiveThreshold(
                norm_img,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                11,
                2
            ),

        "Otsu":
            cv2.threshold(
                norm_img,
                0,
                255,
                cv2.THRESH_BINARY,
                cv2.THRESH_OTSU
            )[1],

        "Manual 100":
            np.where(
                norm_img > 100,
                255,
                0
            ).astype("uint8")
    }

    for key, img in methods.items():

        # 이미지 중심 계산
        if np.sum(img > 0) == 0:
            continue

        cy, cx = center_of_mass(img)

        shift_y = int(
            round(img.shape[0] // 2 - cy)
        )

        shift_x = int(
            round(img.shape[1] // 2 - cx)
        )

        shifted = shift(
            img,
            shift=(shift_y, shift_x),
            mode="constant",
            cval=0
        )

        # 28x28로 변경
        resized = cv2.resize(
            shifted.astype("uint8"),
            (28, 28),
            interpolation=cv2.INTER_AREA
        )

        # 0~1 정규화
        norm = (
            resized.astype("float32") / 255.0
        )

        # PyTorch CNN 입력 형태:
        # [Batch, Channel, Height, Width]
        tensor = torch.from_numpy(
            norm
        ).unsqueeze(0).unsqueeze(0)

        results[key] = {
            "processed": resized,
            "tensor": tensor
        }

    return results


# ------------------------------------------------------------
# 예측 함수
# ------------------------------------------------------------
def predict_tensor(tensor):

    tensor = tensor.to(DEVICE)

    with torch.no_grad():

        output = model(tensor)

        probability = torch.softmax(
            output,
            dim=1
        )[0]

        prediction = int(
            probability.argmax().item()
        )

        confidence = float(
            probability.max().item()
        )

    return prediction, confidence, probability.cpu().numpy()


# ------------------------------------------------------------
# 예측 실행
# ------------------------------------------------------------
if (
    st.button("예측 실행")
    and canvas_result.image_data is not None
    and model is not None
):

    img = canvas_result.image_data[:, :, 0]

    img = Image.fromarray(
        img.astype("uint8")
    ).convert("L")

    img = img.resize((28, 28))

    arr = np.array(img)

    # 입력 이미지 저장
    save_path = os.path.join(
        MODEL_DIR,
        "last_input.png"
    )

    img.save(save_path)

    st.info(
        f"입력 이미지를 저장했습니다: {save_path}"
    )

    # --------------------------------------------------------
    # 입력 히트맵
    # --------------------------------------------------------
    st.subheader("입력 히트맵")

    fig, ax = plt.subplots()

    ax.imshow(
        arr,
        cmap="hot"
    )

    ax.axis("off")

    st.pyplot(fig)

    # --------------------------------------------------------
    # 다중 전처리
    # --------------------------------------------------------
    results = apply_preprocessing(arr)

    if not results:

        st.error(
            "숫자 영역을 찾지 못했습니다."
        )

    else:

        prediction_results = {}

        for method, data in results.items():

            pred, confidence, probability = (
                predict_tensor(data["tensor"])
            )

            prediction_results[method] = {
                "processed": data["processed"],
                "prediction": pred,
                "confidence": confidence,
                "prob": probability
            }

        # 가장 높은 confidence를 가진 결과 선택
        best_method, best = max(
            prediction_results.items(),
            key=lambda x: x[1]["confidence"]
        )

        st.subheader(
            f"최종 예측: **{best['prediction']}** "
            f"(신뢰도: {best['confidence']:.2f})"
        )

        st.caption(
            f"선택된 전처리: {best_method}"
        )

        st.bar_chart(
            best["prob"]
        )

        # ----------------------------------------------------
        # 전처리별 결과
        # ----------------------------------------------------
        st.subheader(
            "다중 전처리 예측 결과"
        )

        for method, data in prediction_results.items():

            st.markdown(
                f"### {method} "
                f"(예측: {data['prediction']}, "
                f"신뢰도: {data['confidence']:.2f})"
            )

            st.image(
                data["processed"],
                width=140
            )

elif model is None:

    st.warning(
        "PyTorch 모델(.pth)이 없습니다. "
        "먼저 CNN 모델을 학습해주세요."
    )
