
import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO
import av
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

# -------------------------------------------------------------------
# 1. 페이지 기본 설정 및 타이틀 출력
# -------------------------------------------------------------------
# Streamlit 웹 페이지의 브라우저 탭 제목과 레이아웃 구조를 설정합니다.
st.set_page_config(
    page_title="YOLOv8 마스크 탐지", 
    layout="centered"  # 중앙 정렬 레이아웃 적용
)

# 메인 화면에 크게 표시될 앱의 메인 타이틀을 출력합니다.
st.title("😷 마스크 착용 상태 탐지 - YOLOv8")

# -------------------------------------------------------------------
# 2. 모델 로드 (캐싱을 통해 중복 로딩 방지)
# -------------------------------------------------------------------
# @st.cache_resource: Streamlit의 캐싱 데코레이터입니다.
# 페이지가 리로드되더라도 YOLOv8 모델을 매번 메모리에 다시 올리지 않고
# 캐시된 모델 인스턴스를 재사용하여 앱 동작 속도를 향상시킵니다.
@st.cache_resource
def load_model():
    # 학습된 마스크 탐지 커스텀 파라미터 파일(best.pt)을 로드합니다.
    # 실행 경로에 'best.pt' 파일이 위치해 있어야 합니다.
    return YOLO("best.pt")

# 정의한 캐싱 함수를 호출하여 로드된 모델 객체를 변수에 저장합니다.
model = load_model()

# -------------------------------------------------------------------
# 3. 마스크 탐지 공통 추론 함수
# -------------------------------------------------------------------
def detect_image(image_bgr):
    """
    입력된 Open-CV 형태(BGR 채널 구조)의 단일 이미지 프레임에 대해 
    YOLOv8 객체 탐지를 수행하고, 경계 상자(Bounding Box)와 클래스 라벨이 
    시각화된 이미지 프레임을 반환하는 공통 함수입니다.
    """
    # 1) YOLOv8 모델에 BGR 이미지 전달하여 추론 수행
    results = model(image_bgr)

    # 2) results[0].plot(): 추론 결과(객체 위치, 신뢰도 score, 클래스명 등)가 
    #    이미지 상에 색상 박스로 렌더링된 새로운 BGR NumPy 배열을 리턴합니다.
    return results[0].plot()

# -------------------------------------------------------------------
# 4. 사이드바 탐지 모드 선택 UI
# -------------------------------------------------------------------
# 사이드바 영역에 3가지 기능 모드를 제공하는 라디오 버튼을 생성합니다.
mode = st.sidebar.radio("탐지 모드 선택", ["이미지", "웹캠", "동영상"])

# ===================================================================
# MODE 1: 단일 이미지 업로드 및 객체 탐지
# ===================================================================
if mode == "이미지":
    # 사용자로부터 이미지 파일(jpg, jpeg, png)을 선택받는 업로더 컴포넌트 생성
    uploaded_file = st.file_uploader("이미지를 업로드하세요", type=["jpg", "jpeg", "png"])

    # 사용자가 파일 업로드를 완료한 경우 실행
    if uploaded_file:
        # 업로드된 메모리 내 파일(BytesIO)을 uint8 바이트 배열로 변환
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)

        # OpenCV를 사용하여 바이트 데이터 배열을 BGR 이미지 행렬 데이터로 디코딩
        image_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        # [원본 이미지 출력]: OpenCV는 기본 BGR 구조를 사용하므로 
        # Streamlit 화면 출력을 위해 RGB 구조로 채널을 변경하여 출력합니다.
        st.image(
            cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB), 
            caption="원본 이미지", 
            use_container_width=True
        )

        # [객체 탐지 및 결과 출력]
        st.subheader("탐지 결과")

        # 추론 함수 호출하여 결과 BGR 이미지 획득
        result_bgr = detect_image(image_bgr)

        # BGR -> RGB 채널 변환 후 탐지 결과 이미지 출력
        st.image(
            cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), 
            caption="탐지된 이미지", 
            use_container_width=True
        )

# ===================================================================
# MODE 2: 실시간 웹캠 스트리밍 탐지 (WebRTC 방식)
# ===================================================================
elif mode == "웹캠":
    # streamlit-webrtc의 최신 규격인 VideoProcessorBase를 상속받는 프레임 처리 클래스 정의
    class YOLOVideoProcessor(VideoProcessorBase):
        # recv 메서드는 실시간으로 전달되는 카메라 프레임(av.VideoFrame)을 처리하는 통로입니다.
        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            # 1) 입력 비디오 프레임을 PyAV 객체에서 OpenCV용 NumPy BGR 데이터로 변환
            img = frame.to_ndarray(format="bgr24")

            # 2) YOLOv8 마스크 탐지 추론 실행
            result = detect_image(img)

            # 3) 추론 결과 BGR 이미지를 다시 PyAV 비디오 프레임으로 변환하여 프론트엔드로 송출
            return av.VideoFrame.from_ndarray(result, format="bgr24")

    try:
        # WebRTC 실시간 영상 전송 스트리머 객체 생성 및 실행
        webrtc_streamer(
            key="mask-detect",  # 해당 컴포넌트를 식별하기 위한 고유 키값
            video_processor_factory=YOLOVideoProcessor,  # 앞서 정의한 비디오 처리 클래스 지정
            media_stream_constraints={"video": True, "audio": False},  # 웹캠 비디오 켜기, 오디오 끄기
            rtc_configuration={
                # 클라우드 배포(Streamlit Cloud 등) 시 P2P 통신 네트워크 연결을 보장하기 위한 STUN/TURN 서버 설정
                "iceServers": [
                    {"urls": "stun:stun.l.google.com:19302"},
                    {
                        "urls": "turn:openrelay.metered.ca:80",
                        "username": "openrelayproject",
                        "credential": "openrelayproject"
                    },
                ]
            },
            async_processing=True,  # 영상 입출력을 비동기(Async) 방식으로 처리하여 병목 방지
        )
    except Exception as e:
        # WebRTC 로딩 실패 시 예외 처리 출력
        st.error(f"웹캠 스트리밍 실행 중 오류 발생: {e}")
        st.info("Streamlit Cloud 환경에서는 TURN/STUN 연결 문제로 웹캠 스트리밍이 실패할 수 있습니다. 이미지 업로드 모드를 권장합니다.")

# ===================================================================
# MODE 3: 동영상 파일 업로드 및 프레임별 연속 재생 탐지
# ===================================================================
elif mode == "동영상":
    # 동영상 파일(mp4, mov, avi) 입력받는 파일 업로더 생성
    uploaded_video = st.file_uploader("동영상을 업로드하세요", type=["mp4", "mov", "avi"])

    if uploaded_video:
        # OpenCV의 cv2.VideoCapture는 바이트 객체를 직접 읽지 못하므로
        # Temp 파일 형태로 디스크에 안전하게 임시 저장 후 로드합니다.
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tfile:
            tfile.write(uploaded_video.read())
            temp_path = tfile.name

        # 임시 저장된 파일 경로로 OpenCV VideoCapture 객체 생성
        cap = cv2.VideoCapture(temp_path)

        # 동영상 프레임이 실시간으로 교체되며 재생될 Streamlit 빈 자리(Placeholder) 생성
        stframe = st.empty()
        st.subheader("탐지 결과 (실시간 재생)")

        try:
            # 동영상의 프레임이 남아있는 동안 무한 루프 수행
            while cap.isOpened():
                # 한 프레임씩 영상 읽기 (ret: 성공 여부, frame: BGR 프레임 이미지)
                ret, frame = cap.read()

                # 영상이 종료되었거나 더 이상 읽을 프레임이 없으면 루프 탈출
                if not ret:
                    break

                # 읽어온 현재 프레임에 마스크 탐지 추론 적용
                result_bgr = detect_image(frame)

                # BGR -> RGB 채널 변환 후 생성해둔 stframe 자리에 실시간으로 이미지를 계속 교체(렌더링)
                stframe.image(
                    cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB), 
                    channels="RGB", 
                    use_container_width=True
                )
        finally:
            # 영상 처리가 끝나거나 오류 발생 시 메모리 자원 해제
            cap.release()

            # 서버 디스크 누수를 방지하기 위해 생성했던 임시 동영상 파일 완전 삭제
            if os.path.exists(temp_path):
                os.remove(temp_path)
