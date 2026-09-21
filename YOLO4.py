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


st.set_page_config(
    page_title="手勢控制錄影機",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 手勢控制 GIF 錄影系統")
st.markdown("比出 **`start`** 手勢即可觸發：**倒數 2 秒 ➔ 自動錄影 3 秒 ➔ 下載 GIF**")


if "gif_data" not in st.session_state:
    st.session_state.gif_data = None

if "record_status" not in st.session_state:
    st.session_state.record_status = "WAITING"


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
        self.state = "WAITING"
        self.timer_start = 0
        self.output_frames = []
        self.gif_data = None
        self.lock = threading.Lock()
        self.last_detection = False

    def video_frame_callback(self, frame):

        img = frame.to_ndarray(format="bgr24")

        img = cv2.resize(img, (640, 480))

        current_time = time.time()

        with self.lock:

            if self.state == "WAITING":

                results = model(
                    img,
                    conf=conf_threshold,
                    verbose=False,
                    classes=[0]
                )

                detected = False

                for result in results:

                    for box in result.boxes:

                        x1, y1, x2, y2 = map(
                            int,
                            box.xyxy[0]
                        )

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
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 255, 0),
                            2
                        )

                        detected = True
                        break

                    if detected:
                        break

                if detected:

                    self.state = "COUNTDOWN"
                    self.timer_start = current_time

                cv2.putText(
                    img,
                    "Show 'start' gesture",
                    (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 0),
                    2
                )

            elif self.state == "COUNTDOWN":

                elapsed = current_time - self.timer_start
                remaining = 2.0 - elapsed

                if remaining <= 0:

                    self.state = "RECORDING"
                    self.timer_start = current_time
                    self.output_frames = []

                else:

                    cv2.putText(
                        img,
                        f"Start in: {remaining:.1f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        3
                    )

            elif self.state == "RECORDING":

                elapsed = current_time - self.timer_start
                remaining = 3.0 - elapsed

                if elapsed >= 3.0:

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

                    if len(self.output_frames) % 2 == 0:

                        rgb = cv2.cvtColor(
                            img,
                            cv2.COLOR_BGR2RGB
                        )

                        self.output_frames.append(
                            Image.fromarray(rgb)
                        )

                    cv2.putText(
                        img,
                        f"REC {remaining:.1f}s",
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 0, 255),
                        3
                    )

            elif self.state == "DONE":

                cv2.putText(
                    img,
                    "Recording Complete",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    3
                )

        return av.VideoFrame.from_ndarray(
            img,
            format="bgr24"
        )


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
            "video": True,
            "audio": False
        },
        async_processing=True
    )


with col2:

    st.subheader("📌 系統狀態")

    if ctx.video_processor:

        processor = ctx.video_processor

        with processor.lock:

            state = processor.state
            gif_data = processor.gif_data

        if state == "WAITING":

            st.info(
                "🔍 等待手勢中\n\n"
                "請面向攝影機比出 `start` 手勢"
            )

        elif state == "COUNTDOWN":

            remaining = max(
                0,
                2.0 - (
                    time.time() -
                    processor.timer_start
                )
            )

            st.warning(
                f"⏳ 偵測到手勢！\n\n"
                f"即將開始錄影：**{remaining:.1f} 秒**"
            )

        elif state == "RECORDING":

            remaining = max(
                0,
                3.0 - (
                    time.time() -
                    processor.timer_start
                )
            )

            st.error(
                f"🔴 錄影中...\n\n"
                f"剩餘時間：**{remaining:.1f} 秒**"
            )

        elif state == "DONE":

            st.success(
                "🎉 錄影完成！"
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

                if st.button(
                    "🔄 再錄一次",
                    use_container_width=True
                ):

                    with processor.lock:
                        processor.state = "WAITING"
                        processor.timer_start = 0
                        processor.output_frames = []
                        processor.gif_data = None

                    st.rerun()

    else:

        st.warning(
            "📷 請按下上方的 Start 按鈕開啟攝影機"
        )


st.markdown("---")

st.info(
    "第一次使用時，瀏覽器會要求允許使用攝影機，請選擇「允許」。"
)
