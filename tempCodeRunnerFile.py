import streamlit as st
import tempfile
import sounddevice as sd
from scipy.io.wavfile import write
import os

from utils.prediction import final_prediction

# ------------------ SESSION STATE ------------------
if "result" not in st.session_state:
    st.session_state.result = None

if "recorded_file" not in st.session_state:
    st.session_state.recorded_file = None

# ------------------ UI ------------------
st.title("Voice Emotion Detection (Female Only)")
st.write("App is running...")  # Debug line

st.sidebar.header("Settings")
duration = st.sidebar.slider("Recording Duration (seconds)", 2, 5, 3)

# ------------------ UPLOAD SECTION ------------------
st.header("Upload Audio")

uploaded_file = st.file_uploader(
    "Upload a WAV file",
    type=["wav"]
)

if uploaded_file is not None:
    st.audio(uploaded_file)

    if st.button("Predict from Uploaded Audio", type="primary"):

        with st.spinner("Processing..."):
            try:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
                temp_file.write(uploaded_file.read())
                temp_path = temp_file.name

                result = final_prediction(temp_path)

                st.session_state.result = result
                st.write("DEBUG RESULT:", result)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ------------------ RECORD SECTION ------------------
st.header("Record Voice")

if st.button("Start Recording"):

    fs = 44100
    st.info("Recording...")

    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()

    file_path = "recorded.wav"
    write(file_path, fs, recording)

    st.session_state.recorded_file = file_path

    st.success("Recording complete")
    st.audio(file_path)

# Predict button (separate)
if st.session_state.recorded_file is not None:

    if st.button("Predict from Recorded Audio", type="primary"):

        with st.spinner("Processing..."):
            try:
                result = final_prediction(st.session_state.recorded_file)

                st.session_state.result = result
                st.write("DEBUG RESULT:", result)

            except Exception as e:
                st.error(f"Error during prediction: {e}")

# ------------------ RESULT DISPLAY ------------------
if st.session_state.result is not None:

    st.subheader("Prediction Result")

    if isinstance(st.session_state.result, str) and "Please upload" in st.session_state.result:
        st.error(st.session_state.result)
    else:
        st.metric(
            "Emotion",
            st.session_state.result.replace("Emotion: ", "")
        )

# ------------------ FOOTER ------------------
st.markdown("*Voice Emotion Detection | Gender Filter + Emotion Model*")