import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av

st.set_page_config(page_title="Camera Test")

st.title("📷 攝影機測試")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")
    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="camera-test",
    video_frame_callback=video_frame_callback,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
