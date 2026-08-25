import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import av
import numpy as np
import cv2
import torch
import tempfile
import logging

st.set_page_config(page_title="YOLOv8 마스크 탐지", layout="centered")
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# MPS(애플 실리콘 GPU)가 있으면 사용. CPU 대비 약 5배 빠름
#   cpu 171.1 ms/frame (5.8 FPS)  vs  mps 34.9 ms/frame (28.7 FPS)
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

@st.cache_resource
def load_model():
    m = YOLO("best.pt")  # 반드시 같은 폴더에 best.pt 포함
    m.to(DEVICE)
    return m

model = load_model()

def detect_image(image_bgr):
    # verbose=False: 프레임마다 콘솔 로그가 쌓이는 것을 방지
    results = model(image_bgr, device=DEVICE, verbose=False)
    return results[0].plot()

mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# 이미지 탐지
if mode == "이미지":
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), caption="원본 이미지", use_container_width=True)

        st.subheader("탐지 결과")
        result_bgr = detect_image(image_bgr)
        st.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), caption="탐지된 이미지", use_container_width=True)

# 웹캠 탐지
elif mode == "웹캠":
    # ★ video_processor_factory 에는 recv() 를 구현한 클래스를 넘겨야 한다.
    #   구버전 API인 transform() 을 쓰면 라이브러리가
    #       av.VideoFrame.from_ndarray(self.transform(frame), ...)
    #   처럼 결과를 한 번 더 감싸는데, transform() 이 이미 VideoFrame 을
    #   반환하고 있어서 이중 래핑으로 예외가 난다. 예외가 워커 스레드에서
    #   삼켜지기 때문에 에러 없이 화면만 멈춘 것처럼 보인다.
    class VideoProcessor(VideoProcessorBase):
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")
            result = detect_image(img)
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    try:
      webrtc_streamer(
         key="mask-detect",
         video_processor_factory=VideoProcessor,
         media_stream_constraints={"video": True, "audio": False},
         rtc_configuration={
             "iceServers": [
                 {"urls": "stun:stun.l.google.com:19302"},
                 {
                     "urls": "turn:openrelay.metered.ca:80",
                     "username": "openrelayproject",
                     "credential": "openrelayproject"
                 },
             ]
         },
         async_processing=True,
      )
    except Exception as e:
      st.error(f"웹캠 스트리밍 실행 중 오류 발생: {e}")
      st.info("Streamlit Cloud 환경에서는 TURN/STUN 연결 문제로 웹캠 스트리밍이 실패할 수 있습니다. 이미지 업로드 모드를 권장합니다.")


# 동영상 탐지
elif mode == "동영상":
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])
    if uploaded_video:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())
        cap = cv2.VideoCapture(tfile.name)

        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            result_bgr = detect_image(frame)
            stframe.image(cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
            # 잠시 대기 - 너무 빠른 루프 방지
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()



