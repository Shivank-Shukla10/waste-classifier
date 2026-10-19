import time
import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="WasteVision AI",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
<style>

.main-title {
    font-size: 42px;
    font-weight: 700;
    margin-bottom: 0;
}

.subtitle {
    font-size: 18px;
    color: #888;
    margin-bottom: 25px;
}

.result-box {
    padding: 22px;
    border-radius: 15px;
    border: 1px solid rgba(46, 125, 50, 0.35);
    background: rgba(46, 125, 50, 0.10);
    margin-top: 15px;
    margin-bottom: 20px;
}

.result-label {
    font-size: 16px;
    color: #999;
    margin-bottom: 8px;
}

.result-value {
    font-size: 34px;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-confidence {
    font-size: 19px;
}

</style>
""",
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">♻️ WasteVision AI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'AI-powered waste classification using MobileNetV2'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Model Information")

    st.write("**Architecture:** MobileNetV2")
    st.write("**Task:** Waste Image Classification")
    st.write("**Classes:** 12")
    st.write("**Input Size:** 160 × 160")

    st.divider()

    st.subheader("📊 Model Performance")

    st.metric(
        "Test Accuracy",
        "92.18%"
    )

    st.metric(
        "Macro F1",
        "89.34%"
    )

    st.divider()

    st.subheader("🛠️ Technology")

    st.write("🐍 Python")
    st.write("🔥 PyTorch")
    st.write("⚡ FastAPI")
    st.write("🎨 Streamlit")
    st.write("📈 MLflow")
    st.write("🐳 Docker")

    st.divider()

    st.caption(
        "WasteVision AI — End-to-End ML Application"
    )


# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a waste image",
    type=["jpg", "jpeg", "png", "webp"],
    help="Upload an image of waste for classification."
)


# ============================================================
# MAIN APPLICATION
# ============================================================

if uploaded_file:

    image_bytes = uploaded_file.getvalue()

    # --------------------------------------------------------
    # IMAGE + PREDICTION LAYOUT
    # --------------------------------------------------------

    col1, col2 = st.columns(
        [1, 1],
        gap="large"
    )


    # ========================================================
    # LEFT COLUMN — INPUT IMAGE
    # ========================================================

    with col1:

        st.subheader("📷 Input Image")

        st.image(
            image_bytes,
            caption=uploaded_file.name,
            use_container_width=True
        )

        st.caption(
            f"File: {uploaded_file.name}"
        )


    # ========================================================
    # RIGHT COLUMN — AI PREDICTION
    # ========================================================

    with col2:

        st.subheader("🤖 AI Prediction")

        analyze_button = st.button(
            "🔍 Analyze Waste",
            type="primary",
            use_container_width=True
        )


        # ----------------------------------------------------
        # ANALYZE BUTTON
        # ----------------------------------------------------

        if analyze_button:

            files = {
                "file": (
                    uploaded_file.name,
                    image_bytes,
                    uploaded_file.type
                )
            }


            try:

                # ==================================================
                # CALL PREDICTION API
                # ==================================================

                start_time = time.perf_counter()

                response = requests.post(
                    f"{API_URL}/predict",
                    files=files,
                    timeout=60
                )

                latency = (
                    time.perf_counter()
                    - start_time
                )


                # ==================================================
                # PREDICTION SUCCESS
                # ==================================================

                if response.status_code == 200:

                    result = response.json()

                    prediction = result["prediction"]

                    confidence = result["confidence"]

                    top3 = result["top_3_predictions"]


                    # ==================================================
                    # PREDICTION RESULT
                    # ==================================================

                    st.markdown("### 🎯 Prediction Result")

                    result_container = st.container(
                        border=True
                    )

                    with result_container:

                        st.caption(
                            "Predicted Waste Type"
                        )

                        st.markdown(
                            f"# {prediction}"
                        )

                        st.write(
                            f"**Confidence: "
                            f"{confidence * 100:.2f}%**"
                        )

                        st.progress(
                            confidence,
                            text=(
                                f"Confidence: "
                                f"{confidence * 100:.2f}%"
                            )
                        )


                    # ==================================================
                    # API LATENCY
                    # ==================================================

                    st.caption(
                        f"⚡ API response time: "
                        f"{latency:.2f} seconds"
                    )


                    # ==================================================
                    # TOP 3 PREDICTIONS
                    # ==================================================

                    st.subheader(
                        "🏆 Top 3 Predictions"
                    )

                    for rank, item in enumerate(
                        top3,
                        start=1
                    ):

                        class_name = item["class"]

                        score = item["confidence"]

                        st.write(
                            f"**{rank}. {class_name}** "
                            f"— {score * 100:.2f}%"
                        )

                        st.progress(
                            score
                        )


                    # ==================================================
                    # GRAD-CAM
                    # ==================================================

                    st.divider()

                    st.subheader(
                        "🔥 Model Explainability"
                    )

                    st.caption(
                        "Grad-CAM highlights the regions "
                        "that influenced the model's prediction."
                    )


                    # ==================================================
                    # REQUEST GRAD-CAM
                    # ==================================================

                    with st.spinner(
                        "Generating Grad-CAM explanation..."
                    ):

                        explain_response = requests.post(
                            f"{API_URL}/explain",
                            files=files,
                            timeout=60
                        )


                    # ==================================================
                    # DISPLAY GRAD-CAM
                    # ==================================================

                    if explain_response.status_code == 200:

                        st.image(
                            explain_response.content,
                            caption=(
                                "🔥 Grad-CAM — "
                                "Model Attention"
                            ),
                            use_container_width=True
                        )

                        st.success(
                            "Explainability visualization "
                            "generated successfully."
                        )

                    else:

                        st.warning(
                            "⚠️ Grad-CAM could not be generated."
                        )

                        st.caption(
                            f"API response: "
                            f"{explain_response.status_code}"
                        )


                # ==================================================
                # PREDICTION API ERROR
                # ==================================================

                else:

                    st.error(
                        f"❌ Prediction API Error: "
                        f"{response.status_code}"
                    )

                    try:

                        error_detail = response.json()

                        st.json(
                            error_detail
                        )

                    except Exception:

                        st.write(
                            response.text
                        )


            # ======================================================
            # CONNECTION ERROR
            # ======================================================

            except requests.exceptions.ConnectionError:

                st.error(
                    "❌ Cannot connect to FastAPI."
                )

                st.info(
                    "Make sure FastAPI is running on "
                    "http://127.0.0.1:8000"
                )


            # ======================================================
            # TIMEOUT ERROR
            # ======================================================

            except requests.exceptions.Timeout:

                st.error(
                    "⏱️ API request timed out."
                )

                st.info(
                    "The model is taking too long to respond. "
                    "Please try again."
                )


            # ======================================================
            # OTHER ERROR
            # ======================================================

            except Exception as e:

                st.error(
                    f"❌ Something went wrong: {e}"
                )


# ============================================================
# NO IMAGE UPLOADED
# ============================================================

else:

    st.info(
        "👆 Upload a waste image to start classification."
    )

    st.markdown(
        """
### 🚀 How it works

1. 📤 Upload a waste image
2. 🔍 Click **Analyze Waste**
3. 🤖 MobileNetV2 classifies the image
4. 🏆 View the Top-3 predictions
5. 🔥 View Grad-CAM model explanation
"""
    )

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#888;">
        <p>WasteVision AI • MobileNetV2 • FastAPI • PyTorch • MLflow</p>
        <p>Model trained on 12 waste categories</p>
    </div>
    """,
    unsafe_allow_html=True
)