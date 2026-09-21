# -- coding: utf-8 --
import streamlit as st
from ultralytics import YOLO
import cv2
import time
from PIL import Image
import io

# ===== 1. 網頁頁面配置 =====
st.set_page_config(
    page_title="手勢控制錄影機",
    page_icon="🎥",
    layout="wide"
)

st.title("🎥 手勢控制 GIF 錄影系統")
st.markdown("比出 **`start`** 手勢即可觸發：**倒數 2 秒 ➔ 自動錄影 3 秒 ➔ 下載 GIF**")

# ===== 2. 側邊欄設定 =====
st.sidebar.header("⚙️ 系統設定")
model_path = st.sidebar.text_input(
    "YOLO 模型路徑", 
    "best.pt"
)
conf_threshold = st.sidebar.slider("辨識信心度 (Confidence)", 0.1, 1.0, 0.2, 0.05)

# 載入模型（使用快取避免重複載入）
@st.cache_resource
def load_yolo_model(path):
    return YOLO(path)

try:
    model = load_yolo_model(model_path)
    st.sidebar.success("✅ 模型載入成功！")
except Exception as e:
    st.sidebar.error(f"❌ 模型載入失敗，請檢查路徑：{e}")
    st.stop()

labels = ["start"]

# ===== 3. 主畫面版面配置 =====
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📹 攝影機即時畫面")
    video_placeholder = st.empty()

with col2:
    st.subheader("📌 系統狀態")
    status_box = st.empty()
    download_box = st.empty()

# 啟動按鈕
start_button = st.button("🚀 開啟攝影機並開始偵測", type="primary", use_container_width=True)

# ===== 4. 主要邏輯 =====
if start_button:
    cap = cv2.VideoCapture(1)

    if not cap.isOpened():
        status_box.error("❌ 無法開啟攝影機，請確認設備連接。")
        st.stop()

    output_frames = []
    frame_count = 0
    state = "WAITING"
    timer_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            status_box.error("無法讀取攝影機畫面。")
            break

        frame = cv2.resize(frame, (480, 360))
        frame_count += 1
        current_time = time.time()
        
        display_frame = frame.copy()

        # --- 狀態 1：等待手勢 ---
        if state == "WAITING":
            status_box.info("🔍 **等待手勢中**\n請面向攝影機比出 `start` 手勢...")
            cv2.putText(display_frame, "Show 'start' gesture", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            if frame_count % 2 == 0:
                results = model(frame, conf=conf_threshold, verbose=False, classes=[0])
                detected = False
                for result in results:
                    for box in result.boxes:
                        # 畫出框線與標籤
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(display_frame, "start", (x1, y1 - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        detected = True
                        break
                    if detected:
                        break

                if detected:
                    state = "COUNTDOWN"
                    timer_start = current_time

        # --- 狀態 2：倒數 2 秒 ---
        elif state == "COUNTDOWN":
            elapsed = current_time - timer_start
            remaining = 2.0 - elapsed

            if remaining <= 0:
                state = "RECORDING"
                timer_start = current_time
            else:
                status_box.warning(f"⏳ **偵測到手勢！**\n即將開始錄影：**{remaining:.1f} 秒**")
                cv2.putText(display_frame, f"Start in: {remaining:.1f}s", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

        # --- 狀態 3：錄影 3 秒 ---
        elif state == "RECORDING":
            elapsed = current_time - timer_start
            rec_remaining = 3.0 - elapsed

            if elapsed >= 3.0:
                status_box.success("🎉 **錄影完成！** 正在生成 GIF...")
                break
            else:
                status_box.error(f"🔴 **錄影中...**\n剩餘時間：**{rec_remaining:.1f} 秒**")
                if frame_count % 2 == 0:
                    gif_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    output_frames.append(Image.fromarray(gif_frame))

                cv2.putText(display_frame, f"REC ({rec_remaining:.1f}s)", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

        # 畫面更新至網頁
        display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
        video_placeholder.image(display_frame_rgb, channels="RGB", use_container_width=True)

    cap.release()

    # ===== 5. 生成 GIF 並提供下載按鈕 =====
    if len(output_frames) > 0:
        gif_bytes = io.BytesIO()
        output_frames[0].save(
            gif_bytes,
            format="GIF",
            save_all=True,
            append_images=output_frames[1:],
            duration=200,
            loop=0
        )
        gif_data = gif_bytes.getvalue()

        st.balloons()  # 播放彩帶慶祝特效
        download_box.image(gif_data, caption="🎬 生成的 GIF 預覽", use_container_width=True)
        download_box.download_button(
            label="💾 下載 GIF 檔案",
            data=gif_data,
            file_name=f"gesture_record_{int(time.time())}.gif",
            mime="image/gif",
            type="primary"
        )
