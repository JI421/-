import streamlit as st
from streamlit_webrtc import webrtc_streamer
from ultralytics import YOLO
import av
import cv2
import threading
import time

st.set_page_config(
    page_title="YOLO 手勢偵測",
    page_icon="🎥"
)

st.title("🎥 YOLO 手勢偵測測試")

@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

lock = threading.Lock()

latest_frame = None
latest_boxes = []
running = True


def detect_loop():
    global latest_frame
    global latest_boxes
    global running

    while running:

        with lock:
            if latest_frame is None:
                frame = None
            else:
                frame = latest_frame.copy()

        if frame is None:
            time.sleep(0.1)
            continue

        try:

            results = model.predict(
                frame,
                conf=0.2,
                imgsz=320,
                classes=[0],
                device="cpu",
                verbose=False
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

            with lock:
                latest_boxes = boxes

        except Exception:

            with lock:
                latest_boxes = []

        time.sleep(0.1)


thread = threading.Thread(
    target=detect_loop,
    daemon=True
)

thread.start()


def video_frame_callback(frame):

    global latest_frame

    img = frame.to_ndarray(format="bgr24")

    with lock:
        latest_frame = img.copy()
        boxes = list(latest_boxes)

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
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

    cv2.putText(
        img,
        "YOLO Detecting...",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2
    )

    return av.VideoFrame.from_ndarray(
        img,
        format="bgr24"
    )


webrtc_streamer(
    key="yolo-test",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
