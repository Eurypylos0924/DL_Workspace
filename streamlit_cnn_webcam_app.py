# ============================================================
# 웹캠 숫자 인식기 - PyTorch (Canny Edge + 정밀 윤곽선 추출)
# ============================================================

import os
import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps
import torch
import torch.nn as nn

# ------------------------------------------------------------
# 모델 설정 및 정의
# ------------------------------------------------------------
MODEL_DIR = "saved_models"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

def get_latest_model():
    if not os.path.exists(MODEL_DIR):
        return None
    models = [f for f in os.listdir(MODEL_DIR) if f.endswith(".pth")]
    if not models:
        return None
    models.sort(reverse=True)
    return os.path.join(MODEL_DIR, models[0])

# 모델 로드
model_path = get_latest_model()
model = None

if model_path:
    model = MNISTCNN().to(DEVICE)
    model.load_state_dict(
        torch.load(model_path, map_location=DEVICE, weights_only=True)
    )
    model.eval()

# ------------------------------------------------------------
# 정밀 전처리 (Canny Edge + 엣지 팽창 + 노이즈 제거)
# ------------------------------------------------------------
def preprocess_image(gray_arr):
    h_orig, w_orig = gray_arr.shape

    # 1. 외곽 8% 마진 마스킹
    margin_h = int(h_orig * 0.08)
    margin_w = int(w_orig * 0.08)

    # 2. 가우시안 블러로 종이 질감/접힘선 평탄화
    blurred = cv2.GaussianBlur(gray_arr, (7, 7), 0)

    # 3. Canny Edge 추출 (펜 선의 경계선만 선명히 검출)
    edges = cv2.Canny(blurred, 50, 150)

    # 마진 테두리 강제 제거
    edges[:margin_h, :] = 0
    edges[-margin_h:, :] = 0
    edges[:, :margin_w] = 0
    edges[:, -margin_w:] = 0

    # 4. Dilate를 적용하여 Canny 엣지의 틈을 채우고 굵게 만듦
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    binary = cv2.dilate(edges, kernel, iterations=2)

    # 5. 윤곽선 탐지
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None, None, binary

    # 6. 진짜 숫자 윤곽선 선별 (상단/하단 종이 구김선 제외)
    valid_contours = []
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        aspect_ratio = w / float(h)

        # 선 형태나 너무 얇거나 가로로 긴 구김선(aspect_ratio > 3.0) 제외
        if area > 150 and w > 12 and h > 20 and aspect_ratio < 2.5:
            valid_contours.append(c)

    if not valid_contours:
        return None, None, binary

    # 가장 큰 숫자 윤곽선 선택
    c = max(valid_contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)

    # 7. 숫자 영역 Crop
    digit_crop = binary[y:y+h, x:x+w]

    # 8. 비율 유지 리사이즈 (20x20 크기)
    if h > w:
        new_h = 20
        new_w = max(1, int(round(w * (20.0 / h))))
    else:
        new_w = 20
        new_h = max(1, int(round(h * (20.0 / w))))

    resized_digit = cv2.resize(digit_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 9. 28x28 캔버스 패딩
    padded_28x28 = np.zeros((28, 28), dtype=np.uint8)
    start_y = (28 - new_h) // 2
    start_x = (28 - new_w) // 2
    padded_28x28[start_y:start_y+new_h, start_x:start_x+new_w] = resized_digit

    # 10. MNIST Normalization
    input_arr = padded_28x28.astype("float32") / 255.0
    normalized_arr = (input_arr - 0.1307) / 0.3081

    tensor = torch.from_numpy(normalized_arr).unsqueeze(0).unsqueeze(0)

    return padded_28x28, tensor, binary

# ------------------------------------------------------------
# 예측
# ------------------------------------------------------------
def predict(tensor):
    tensor = tensor.to(DEVICE)
    with torch.no_grad():
        output = model(tensor)
        probability = torch.softmax(output, dim=1)[0]
        pred_class = int(probability.argmax().item())
    return pred_class, probability.cpu().numpy()

# ------------------------------------------------------------
# Streamlit UI
# ------------------------------------------------------------
st.set_page_config(page_title="웹캠 숫자 인식기", layout="centered")
st.title("웹캠 숫자 인식기 (MNIST + PyTorch)")

if model is None:
    st.warning("PyTorch 모델(.pth)이 없습니다. 먼저 모델을 학습해주세요.")
else:
    image_data = st.camera_input("숫자가 잘 보이도록 촬영하세요")

    if image_data is not None:
        image = Image.open(image_data).convert("RGB")
        gray_np = np.array(ImageOps.grayscale(image))

        padded_28x28, tensor, debug_binary = preprocess_image(gray_np)

        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="원본 촬영 이미지", use_container_width=True)

        if padded_28x28 is None:
            st.error("숫자를 찾지 못했습니다. 종이를 펼치고 펜을 약간 더 굵게 써주세요.")
            st.image(debug_binary, caption="전처리 마스크", width=200)
        else:
            pred_class, probability = predict(tensor)

            with col2:
                st.image(padded_28x28, caption="28x28 최종 입력 결과", width=160)

            st.subheader(f"예측된 숫자: **{pred_class}**")
            st.bar_chart(probability)
