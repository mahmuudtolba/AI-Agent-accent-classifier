
import streamlit as st
import requests

# Define the API endpoint
CHECK_MODEL_URL = "http://localhost:8000/api/v1/check_model"
PREDICT_URL = "http://localhost:8000/api/v1/accent_classification"

headers = {
  'Content-Type': 'application/json'
}

st.title("AI Agent Accent Classifier Serving Over REST API")


if st.button("Load Model"):
    try:
        response = requests.get(CHECK_MODEL_URL)
        if response.status_code == 200:
            message = response.json().get("model_exists", "No message returned")
            st.info(f"ℹ️ {message}")
        else:
            st.error(f"⚠️ API returned status code: {response.status_code}")
    except Exception as e:
        st.error(f"🚫 Request failed: {e}")

# --- Step 2: User Inputs Video URL ---
url_input = st.text_input("Enter Video URL (e.g., Google Drive MP4):")



# --- Step 3: Predict ---
if st.button("Predict"):
    if not url_input:
        st.warning("🚨 Please enter a valid video URL.")
    else:
        try:
            payload = {"video_url": [url_input]}
            with st.spinner("Predicting... Please wait!!!"):
                response = requests.post(PREDICT_URL, json=payload)
                if response.status_code == 200:
                    st.success("✅ Prediction complete:")
                    st.json(response.json())
                else:
                    st.error(f"⚠️ Prediction failed: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Request failed: {e}")