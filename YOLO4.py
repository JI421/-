# -- coding: utf-8 --

import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
from ultralytics import YOLO
import cv2
import time
import threading
import io
from PIL import Image
import av

st.set_page_config(
    page_title="手勢控制錄影機",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 手勢控制 GIF 錄影系統")
st.markdown("比出 **start** 手勢即可觸發：**倒數 2 秒 ➔ 自動錄影 3 秒 ➔ 產生 GIF**")

# ===== 1. 側邊欄設定 =====
st.sidebar.header("⚙️ 系統設定")
model_path = st.sidebar.text_input("YOLO 模型路徑", "best.pt")
conf_threshold = st.sidebar.slider("辨識信心度 (Confidence)", 0.1, 1.0, 0.2, 0.05)

@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

try:
    model = load_yolo_model(model_path)
    st.sidebar.success("✅ 模型載入成功！")
except Exception as e:
    st.sidebar.error(f"❌ 模型載入失敗：{e}")
    st.stop()

# ===== 2. WebRTC STUN 伺服器設定 (解決一直轉圈圈的核心關鍵) =====
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ===== 3. WebRTC 影像處理器 =====
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.lock = threading.Lock()
        self.state = "WAITING"
        self.timer_start = 0
        self.latest_frame = None
        self.last_boxes = []
        self.output_frames = []
        self.gif_data = None
        self.running = True
        self.frame_count = 0
        self.last_record_time = 0
        self.fps = 8

        # 啟動背景 YOLO 偵測線程
        self.detection_thread = threading.Thread(
            target=self.detection_loop,
            daemon=True
        )
        self.detection_thread.start()

    def detection_loop(self):
        """背景獨立執行 YOLO 推論，避免卡住 WebRTC 主畫面"""
        while self.running:
            frame = None
            with self.lock:
                if self.latest_frame is not None:
                    frame = self.latest_frame.copy()
                state = self.state

            if frame is None or state != "WAITING":
                time.sleep(0.1)
                continue

            try:
                results = model.predict(
                    frame,
                    conf=conf_threshold,
                    imgsz=256,
                    classes=[0],
                    device="cpu",
                    verbose=False
                )
                boxes = []
                for result in results:
                    for box in result.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        boxes.append((x1, y1, x2, y2))

                with self.lock:
                    self.last_boxes = boxes
            except Exception:
                with self.lock:
                    self.last_boxes = []

            time.sleep(0.15)

    def make_gif_async(self):
        """改為背景線程製作 GIF，不卡死 recv()"""
        def task():
            with self.lock:
                frames = list(self.output_frames)

            if len(frames) == 0:
                with self.lock:
                    self.state = "DONE"
                return

            gif_bytes = io.BytesIO()
            frames[0].save(
                gif_bytes,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=125,
                loop=0
            )

            with self.lock:
                self.gif_data = gif_bytes.getvalue()
                self.state = "DONE"

        threading.Thread(target=task, daemon=True).start()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        current_time = time.time()
        self.frame_count += 1

        # 每 3 幀取 1 幀給背景偵測
        if self.frame_count % 3 == 0:
            with self.lock:
                self.latest_frame = img.copy()

        with self.lock:
            state = self.state
            boxes = list(self.last_boxes)
            timer_start = self.timer_start

        # --- 狀態處理 ---
        if state == "WAITING":
            for x1, y1, x2, y2 in boxes:
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, "start", (x1, max(30, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.putText(img, "Show 'start' gesture", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            if len(boxes) > 0:
                with self.lock:
                    if self.state == "WAITING":
                        self.state = "COUNTDOWN"
                        self.timer_start = current_time
                        self.last_boxes = []

        elif state == "COUNTDOWN":
            remaining = 2.0 - (current_time - timer_start)
            if remaining <= 0:
                with self.lock:
                    self.state = "RECORDING"
                    self.timer_start = current_time
                    self.output_frames = []
                    self.last_record_time = 0
            else:
                cv2.putText(img, f"Start in: {remaining:.1f}s", (20, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        elif state == "RECORDING":
            elapsed = current_time - timer_start
            remaining = 3.0 - elapsed

            if current_time - self.last_record_time >= 1.0 / self.fps:
                rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                h, w = rgb.shape[:2]
                target_w = 640
                target_h = int(h * target_w / w)
                small = cv2.resize(rgb, (target_w, target_h))

                with self.lock:
                    self.output_frames.append(Image.fromarray(small))
                self.last_record_time = current_time

            cv2.putText(img, f"REC {max(0, remaining):.1f}s", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            if elapsed >= 3.0:
                with self.lock:
                    self.state = "PROCESSING"  # 進入處理狀態，防止重複觸發
                self.make_gif_async()

        elif state == "PROCESSING":
            cv2.putText(img, "Processing GIF...", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)

        elif state == "DONE":
            cv2.putText(img, "Recording Complete!", (20, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def on_ended(self):
        self.running = False

# ===== 4. 主畫面 UI 配置 =====
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 攝影機即時畫面")
    ctx = webrtc_streamer(
        key="gesture-camera",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,  # 👈 修正轉圈圈的關鍵設定
        video_processor_factory=GestureProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
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
            status_placeholder.info("🔍 **等待手勢中**\n\n請面向攝影機比出 start 手勢")
        elif state == "COUNTDOWN":
            rem = max(0, 2.0 - (time.time() - timer_start))
            status_placeholder.warning(f"⏳ **偵測到手勢！**\n\n即將開始錄影：**{rem:.1f} 秒**")
        elif state == "RECORDING":
            rem = max(0, 3.0 - (time.time() - timer_start))
            status_placeholder.error(f"🔴 **錄影中...**\n\n剩餘時間：**{rem:.1f} 秒**")
        elif state == "PROCESSING":
            status_placeholder.info("⚙️ **正在打包生成 GIF 中...**")
        elif state == "DONE":
            status_placeholder.success("🎉 **錄影完成！**")
            if gif_data:
                st.image(gif_data, caption="🎬 GIF 預覽", use_container_width=True)
                st.download_button(
                    label="💾 下載 GIF",
                    data=gif_data,
                    file_name=f"gesture_record_{int(time.time())}.gif",
                    mime="image/gif",
                    type="primary",
                    use_container_width=True
                )
    else:
        status_placeholder.info("📷 請點擊左側 **SELECT DEVICE / START** 開啟攝影機，並在瀏覽器允許相機權限。")

st.markdown("---")
st.info("💡 提醒：第一次開啟時，請務必在瀏覽器彈出的提示中點擊「允許」使用攝影機。")
