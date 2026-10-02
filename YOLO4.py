# -- coding: utf-8 --

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import io
import threading
import av
import numpy as np


st.set_page_config(
    page_title="手勢控制錄影機",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 手勢控制 GIF 錄影系統")
st.markdown(
    "比出 **start** 手勢即可觸發："
    "**倒數 2 秒 ➔ 自動錄影 3 秒 ➔ 下載 GIF**"
)


if "gif_data" not in st.session_state:
    st.session_state.gif_data = None


st.sidebar.header("⚙️ 系統設定")

model_path = st.sidebar.text_input(
    "YOLO 模型路徑",
    "best.pt"
)

conf_threshold = st.sidebar.slider(
    "辨識信心度 (Confidence)",
    0.1,
    1.0,
    0.2,
    0.05
)


@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)


try:
    model = load_yolo_model(model_path)
    st.sidebar.success("✅ 模型載入成功！")
except Exception as e:
    st.sidebar.error(f"❌ 模型載入失敗：{e}")
    st.stop()


class GestureProcessor(VideoProcessorBase):

    def __init__(self):

        self.lock = threading.Lock()

        self.state = "WAITING"

        self.timer_start = 0

        self.output_frames = []

        self.gif_data = None

        self.latest_frame = None

        self.latest_boxes = []

        self.running = True

        self.inference_thread = threading.Thread(
            target=self.inference_loop,
            daemon=True
        )

        self.inference_thread.start()


    def inference_loop(self):

        while self.running:

            frame = None

            with self.lock:

                if self.latest_frame is not None:

                    frame = self.latest_frame.copy()

            if frame is None:

                time.sleep(0.02)
                continue


            with self.lock:

                state = self.state


            if state != "WAITING":

                time.sleep(0.05)
                continue


            try:

                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    imgsz=320,
                    verbose=False,
                    classes=[0],
                    device="cpu"
                )

                boxes = []

                for result in results:

                    for box in result.boxes:

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

                        boxes.append(
                            (x1, y1, x2, y2)
                        )

                        break

                    if boxes:
                        break


                with self.lock:

                    self.latest_boxes = boxes

                    if boxes and self.state == "WAITING":

                        self.state = "COUNTDOWN"

                        self.timer_start = time.time()

                        self.latest_boxes = []


            except Exception:

                with self.lock:
                    self.latest_boxes = []


            time.sleep(0.05)


    def video_frame_callback(self, frame):

        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()


        with self.lock:

            self.latest_frame = img.copy()

            state = self.state

            boxes = list(self.latest_boxes)

            timer_start = self.timer_start


        if state == "WAITING":

            for x1, y1, x2, y2 in boxes:

                cv2.rectangle(
                    img,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    img,
                    "start",
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )


            cv2.putText(
                img,
                "Show 'start' gesture",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )


        elif state == "COUNTDOWN":

            elapsed = current_time - timer_start

            remaining = 2.0 - elapsed


            if remaining <= 0:

                with self.lock:

                    if self.state == "COUNTDOWN":

                        self.state = "RECORDING"

                        self.timer_start = current_time

                        self.output_frames = []

            else:

                cv2.putText(
                    img,
                    f"Start in: {remaining:.1f}s",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 255),
                    3
                )


        elif state == "RECORDING":

            elapsed = current_time - timer_start

            remaining = 3.0 - elapsed


            if elapsed >= 3.0:

                with self.lock:

                    if len(self.output_frames) > 0:

                        gif_bytes = io.BytesIO()

                        self.output_frames[0].save(
                            gif_bytes,
                            format="GIF",
                            save_all=True,
                            append_images=self.output_frames[1:],
                            duration=100,
                            loop=0
                        )

                        self.gif_data = gif_bytes.getvalue()

                    self.state = "DONE"


            else:

                rgb = cv2.cvtColor(
                    img,
                    cv2.COLOR_BGR2RGB
                )

                small = cv2.resize(
                    rgb,
                    (640, 480)
                )

                with self.lock:

                    if (
                        len(self.output_frames) == 0
                        or time.time() - getattr(
                            self,
                            "last_frame_time",
                            0
                        ) >= 0.1
                    ):

                        self.output_frames.append(
                            Image.fromarray(small)
                        )

                        self.last_frame_time = time.time()


                cv2.putText(
                    img,
                    f"REC {remaining:.1f}s",
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3
                )


        elif state == "DONE":

            cv2.putText(
                img,
                "Recording Complete",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                3
            )


        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


    def stop(self):

        self.running = False

        super().stop()


RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {
                "urls": [
                    "stun:stun.l.google.com:19302"
                ]
            }
        ]
    }
)


col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("📹 攝影機即時畫面")

    ctx = webrtc_streamer(
        key="gesture-camera",
        video_processor_factory=GestureProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280},
                "height": {"ideal": 720},
                "frameRate": {"ideal": 15}
            },
            "audio": False
        },
        async_processing=True
    )


with col2:

    st.subheader("📌 系統狀態")

    status = st.empty()


    if ctx.video_processor:

        processor = ctx.video_processor

        with processor.lock:

            state = processor.state

            gif_data = processor.gif_data


        if state == "WAITING":

            status.info(
                "🔍 **等待手勢中**\n\n"
                "請面向攝影機比出 start 手勢"
            )


        elif state == "COUNTDOWN":

            status.warning(
                "⏳ **偵測到手勢！**\n\n"
                "即將開始錄影"
            )


        elif state == "RECORDING":

            status.error(
                "🔴 **錄影中...**"
            )


        elif state == "DONE":

            status.success(
                "🎉 **錄影完成！**"
            )


            if gif_data:

                st.image(
                    gif_data,
                    caption="🎬 生成的 GIF 預覽",
                    use_container_width=True
                )

                st.download_button(
                    label="💾 下載 GIF 檔案",
                    data=gif_data,
                    file_name=f"gesture_record_{int(time.time())}.gif",
                    mime="image/gif",
                    type="primary",
                    use_container_width=True
                )


    else:

        status.info(
            "📷 請按下 Start 開啟攝影機"
        )


st.markdown("---")

st.info(
    "第一次使用時，請允許瀏覽器使用攝影機。"
)
