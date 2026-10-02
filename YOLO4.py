# -- coding: utf-8 --

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import io
import threading
import av


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

        self.frame_count = 0

        self.last_boxes = []

        self.latest_frame = None

        self.detection_running = True

        self.detection_thread = threading.Thread(
            target=self.detection_loop,
            daemon=True
        )

        self.detection_thread.start()

        self.fps = 10

        self.last_record_time = 0


    def detection_loop(self):

        while self.detection_running:

            frame = None

            with self.lock:

                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()

                state = self.state

            if frame is None:

                time.sleep(0.02)
                continue

            if state == "WAITING":

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
                        self.last_boxes = boxes

                except Exception:

                    with self.lock:
                        self.last_boxes = []

            time.sleep(0.05)


    def video_frame_callback(self, frame):

        img = frame.to_ndarray(format="bgr24")

        current_time = time.time()

        self.frame_count += 1

        with self.lock:

            self.latest_frame = img.copy()

            state = self.state

            boxes = list(self.last_boxes)


        if state == "WAITING":

            detected = len(boxes) > 0

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

            if detected:

                with self.lock:

                    if self.state == "WAITING":

                        self.state = "COUNTDOWN"

                        self.timer_start = current_time

                        self.last_boxes = []


        elif state == "COUNTDOWN":

            elapsed = current_time - self.timer_start

            remaining = 2.0 - elapsed

            if remaining <= 0:

                with self.lock:

                    self.state = "RECORDING"

                    self.timer_start = current_time

                    self.output_frames = []

                    self.last_record_time = 0

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

            elapsed = current_time - self.timer_start

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

                if (
                    current_time - self.last_record_time
                    >= 1.0 / self.fps
                ):

                    rgb = cv2.cvtColor(
                        img,
                        cv2.COLOR_BGR2RGB
                    )

                    small = cv2.resize(
                        rgb,
                        (640, 480)
                    )

                    self.output_frames.append(
                        Image.fromarray(small)
                    )

                    self.last_record_time = current_time

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

        self.detection_running = False


col1, col2 = st.columns([2, 1])


with col1:

    st.subheader("📹 攝影機即時畫面")

    ctx = webrtc_streamer(
        key="gesture-camera",
        video_processor_factory=GestureProcessor,
        media_stream_constraints={
            "video": True,
            "audio": False
        },
        async_processing=True
    )


with col2:

    st.subheader("📌 系統狀態")

    status_placeholder = st.empty()

    if ctx.video_processor:

        processor = ctx.video_processor

        with processor.lock:

            state = processor.state

            gif_data = processor.gif_data

            timer_start = processor.timer_start


        if state == "WAITING":

            status_placeholder.info(
                "🔍 **等待手勢中**\n\n"
                "請面向攝影機比出 start 手勢"
            )


        elif state == "COUNTDOWN":

            remaining = max(
                0,
                2.0 - (
                    time.time() - timer_start
                )
            )

            status_placeholder.warning(
                f"⏳ **偵測到手勢！**\n\n"
                f"即將開始錄影：**{remaining:.1f} 秒**"
            )


        elif state == "RECORDING":

            remaining = max(
                0,
                3.0 - (
                    time.time() - timer_start
                )
            )

            status_placeholder.error(
                f"🔴 **錄影中...**\n\n"
                f"剩餘時間：**{remaining:.1f} 秒**"
            )


        elif state == "DONE":

            status_placeholder.success(
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

        status_placeholder.info(
            "📷 請按下 Start 開啟攝影機"
        )


st.markdown("---")

st.info(
    "第一次使用時，請允許瀏覽器使用攝影機。"
)
