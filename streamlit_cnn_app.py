# ============================================================
# MNIST 테스트 샘플 예측 - PyTorch 버전
# ============================================================

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

import torch
import torch.nn as nn
from torchvision import datasets
import torchvision.transforms.v2 as transform


# ------------------------------------------------------------
# 기본 설정
# ------------------------------------------------------------
MODEL_DIR = "saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
# 최신 PyTorch 모델 찾기
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
# 학습 로그 로드
# ------------------------------------------------------------
def load_training_log(
    log_path="saved_models/training_log.json"
):
    if not os.path.exists(log_path):
        return None

    try:
        with open(
            log_path,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)

    except (json.JSONDecodeError, OSError):
        return None


# ------------------------------------------------------------
# 학습 로그 시각화
# ------------------------------------------------------------
def plot_training_log(log_data):

    st.subheader("학습 로그 (Accuracy / Loss)")

    fig, ax = plt.subplots(
        1,
        2,
        figsize=(10, 4)
    )

    ax[0].plot(
        log_data["accuracy"],
        label="Train Acc"
    )

    ax[0].plot(
        log_data["val_accuracy"],
        label="Val Acc"
    )

    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()
    ax[0].grid(True)

    ax[1].plot(
        log_data["loss"],
        label="Train Loss"
    )

    ax[1].plot(
        log_data["val_loss"],
        label="Val Loss"
    )

    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()
    ax[1].grid(True)

    st.pyplot(fig)


# ------------------------------------------------------------
# 모델 로드
# ------------------------------------------------------------
def load_model():

    model_path = get_latest_model()

    if model_path is None:
        return None, None

    model = MNISTCNN().to(DEVICE)

    model.load_state_dict(
        torch.load(
            model_path,
            map_location=DEVICE,
            weights_only=True
        )
    )

    model.eval()

    return model, model_path


model, latest_model_path = load_model()


# ------------------------------------------------------------
# MNIST 테스트 데이터
# ------------------------------------------------------------
mnist_transform = transform.Compose([
    transform.ToImage(),
    transform.ToDtype(
        torch.float32,
        scale=True
    )
])

test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=mnist_transform
)


# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(
    page_title="MNIST Test Sample Prediction",
    layout="centered"
)

st.title("CNN 숫자 예측기 (MNIST 샘플 선택)")

st.markdown(
    "`X_test`의 실제 손글씨 샘플을 선택하여 "
    "PyTorch CNN 모델이 예측합니다."
)


# ------------------------------------------------------------
# 학습 로그
# ------------------------------------------------------------
log_data = load_training_log()

if log_data:
    plot_training_log(log_data)
else:
    st.info(
        "학습 로그 파일이 없거나 비어 있습니다."
    )


# ------------------------------------------------------------
# 테스트 샘플 선택 및 예측
# ------------------------------------------------------------
if model is not None:

    st.markdown("### 테스트 샘플 선택")

    sample_index = st.slider(
        "샘플 인덱스 선택",
        min_value=0,
        max_value=len(test_dataset) - 1,
        value=0
    )

    image, label = test_dataset[sample_index]

    st.image(
        image.squeeze(0).numpy(),
        caption=f"실제 숫자: {label}",
        width=150
    )

    # --------------------------------------------------------
    # 예측
    # --------------------------------------------------------
    with torch.no_grad():

        output = model(
            image.unsqueeze(0).to(DEVICE)
        )

        probability = torch.softmax(
            output,
            dim=1
        )[0]

        pred_class = int(
            probability.argmax().item()
        )

    st.subheader(
        f"예측된 숫자: **{pred_class}**"
    )

    st.bar_chart(
        probability.cpu().numpy()
    )

else:

    st.warning(
        "PyTorch 모델(.pth)이 없습니다. "
        "먼저 학습을 완료해주세요."
    )
