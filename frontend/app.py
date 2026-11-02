import streamlit as st
from pathlib import Path
import tempfile, os, sys

HERE = Path(__file__).resolve().parent
PROJECT_DIR = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(PROJECT_DIR))

from inference import FusionVideoPredictor

st.set_page_config(page_title="UCF101 RGB + Pose Recognition", page_icon="🎬", layout="wide")
st.title("UCF101 Human Action Recognition")
st.caption("RGB + tracked-pose late-fusion model")

@st.cache_resource(show_spinner="Loading trained fusion model and vision models...")
def load_predictor():
    return FusionVideoPredictor(PROJECT_DIR)

with st.sidebar:
    st.subheader("Model files")
    st.write("Fusion checkpoint:")
    st.code(str(PROJECT_DIR / "Notebook_2_Improved_v2" / "outputs" / "models"))
    st.write("ViTPose checkpoint:")
    st.code(str(PROJECT_DIR / "checkpoints" / "epoch_209.pth"))
    st.caption("The first run may download the pretrained R3D-18 Kinetics-400 weights if they are not cached.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv"])
if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        video_path = Path(tmp.name)
    st.markdown(
    """
    <style>
    video {
        width: 400px !important;
        max-width: 100% !important;
        height: 400px !important;
        display: block;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
    )

    st.video(video_path)

    if st.button("Predict action", type="primary"):
        try:
            predictor = load_predictor()
            with st.spinner("Detecting and tracking the person, estimating pose, extracting RGB features, and predicting..."):
                result = predictor.predict(video_path)
            st.success("Prediction completed")
            a,b,c = st.columns(3)
            a.metric("Predicted action", result["label"])
            b.metric("Confidence", f'{result["confidence"]:.1%}')
            c.metric("Pose windows", result["num_windows"])
            st.subheader("Video-level prediction")
            st.write(f"**{result['label']}** — {result['confidence']:.1%} confidence")
            st.subheader("Top 5 action scores")
            st.bar_chart(result["top5"].set_index("Action"))
            st.info(f"Dominant tracked person ID: {result['track_id']}. This model was trained on one dominant-person pose sequence per video; it does not provide separately trained predictions for every person in a multi-person scene.")
            st.caption(f"Device: {result['device']} | Frames decoded: {result['decoded_frames']} | Valid pose frames: {result['valid_pose_frames']}")
        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
            st.exception(exc)
        finally:
            try: video_path.unlink(missing_ok=True)
            except Exception: pass
else:
    st.write("Upload a UCF101-style video to classify its action.")
