"""
Valve Leak Detection - Cloud Demo
Streamlit app for Monday proof-of-concept demonstration
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xml_feature_extractor import extract_features_from_xml, parse_curves_xml, get_curve_info
import os

# Page configuration
st.set_page_config(
    page_title="Valve Leak Detection Demo",
    page_icon="🔧",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        padding: 2rem;
        border-radius: 10px;
        text-align: center;
        font-size: 2rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .leak-detected {
        background-color: #ffebee;
        color: #c62828;
        border: 3px solid #c62828;
    }
    .normal-detected {
        background-color: #e8f5e9;
        color: #2e7d32;
        border: 3px solid #2e7d32;
    }
    .confidence-text {
        font-size: 1.5rem;
        margin-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Load model and scaler
@st.cache_resource
def load_model():
    """Load pre-trained model and scaler"""
    try:
        model_path = 'leak_detector_model.pkl'
        scaler_path = 'feature_scaler.pkl'

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)

        return model, scaler
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None

# Header
st.markdown('<div class="main-header">Valve Leak Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Proof-of-Concept Demo | Acoustic Emission Analysis</div>', unsafe_allow_html=True)

# Introduction
with st.expander("ℹ️ About This Demo", expanded=False):
    st.markdown("""
    ### What This Demo Does

    This proof-of-concept demonstrates **AI-powered valve leak detection** using acoustic emission (AE) sensor data.

    **How it works:**
    1. Upload a **Curves XML file** containing AE sensor readings (36-44 kHz ultrasonic)
    2. System extracts 8 statistical features from the waveform
    3. Machine learning model (Random Forest) predicts: **LEAK** or **NORMAL**
    4. Shows confidence score and detailed analysis

    **Model Performance:**
    - 81.8% accuracy on test data
    - **100% leak recall** (all leaks detected)
    - Trained on 50 leak + 109 normal valve samples
    - 96.8% AE/Ultrasonic sensor data

    **Note:** This is a basic demo with 8 simple features. The full system (Week 2-4 pilot) will use:
    - 20 engineered features (statistical + spectral + temporal)
    - Ensemble models (XGBoost + Random Forest)
    - Target: 85-88% accuracy
    """)

st.markdown("---")

# File uploader
st.subheader("📁 Upload Valve Data (XML File)")
uploaded_file = st.file_uploader(
    "Choose a Curves XML file",
    type=['xml'],
    help="Upload a Windrock Curves XML file containing AE sensor readings"
)

if uploaded_file is not None:
    # Read XML content
    xml_content = uploaded_file.read().decode('utf-8')

    # Display file info
    st.success(f"✅ File uploaded: **{uploaded_file.name}**")

    # Get curve metadata
    with st.spinner("Analyzing XML file..."):
        curve_info = get_curve_info(xml_content)

    if 'error' not in curve_info:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Curves", curve_info['total_curves'])
        col2.metric("AE Curves Found", len(curve_info['ae_curves']))
        col3.metric("Data Points", curve_info['data_points'])
        col4.metric("Crank Angle Range", curve_info['crank_angle_range'])

        if curve_info['ae_curves']:
            st.info(f"🎯 Using AE curve: **{curve_info['ae_curves'][0]}**")
        else:
            st.warning("⚠️ No AE/Ultrasonic curve found. Using first available curve.")

    st.markdown("---")

    # Analyze button
    if st.button("🔍 Analyze Valve", type="primary", use_container_width=True):
        with st.spinner("Extracting features and running AI model..."):
            # Extract features
            features = extract_features_from_xml(xml_content)

            if features is None:
                st.error("❌ Failed to extract features from XML file. Please check file format.")
            else:
                # Load model
                model, scaler = load_model()

                if model is None:
                    st.error("❌ Failed to load model. Please contact support.")
                else:
                    # Prepare features for prediction
                    feature_df = pd.DataFrame([features])
                    feature_scaled = scaler.transform(feature_df)

                    # Make prediction
                    prediction = model.predict(feature_scaled)[0]
                    probabilities = model.predict_proba(feature_scaled)[0]

                    # Display result
                    st.markdown("## 🎯 Analysis Result")

                    if prediction == 1:  # Leak
                        st.markdown(
                            f'<div class="result-box leak-detected">⚠️ LEAK DETECTED</div>',
                            unsafe_allow_html=True
                        )
                        confidence = probabilities[1] * 100
                        st.markdown(
                            f'<div class="confidence-text" style="text-align: center;">Confidence: {confidence:.1f}%</div>',
                            unsafe_allow_html=True
                        )

                        st.warning("**Recommendation:** This valve shows leak signatures. Schedule maintenance inspection.")
                    else:  # Normal
                        st.markdown(
                            f'<div class="result-box normal-detected">✅ NORMAL OPERATION</div>',
                            unsafe_allow_html=True
                        )
                        confidence = probabilities[0] * 100
                        st.markdown(
                            f'<div class="confidence-text" style="text-align: center;">Confidence: {confidence:.1f}%</div>',
                            unsafe_allow_html=True
                        )

                        st.success("**Status:** Valve operating within normal parameters.")

                    st.markdown("---")

                    # Visualization and features in columns
                    col_left, col_right = st.columns([3, 2])

                    with col_left:
                        st.subheader("📊 Waveform Visualization")

                        # Parse and plot waveform
                        df = parse_curves_xml(xml_content)
                        if df is not None:
                            # Get first AE curve
                            curve_cols = [col for col in df.columns if col != 'Crank Angle']
                            ae_cols = [col for col in curve_cols
                                      if 'ULTRASONIC' in col.upper() or 'AE' in col.upper() or 'KHZ' in col.upper()]

                            curve_to_plot = ae_cols[0] if ae_cols else curve_cols[0]

                            # Filter outliers for better visualization using IQR method
                            q1 = df[curve_to_plot].quantile(0.25)
                            q3 = df[curve_to_plot].quantile(0.75)
                            iqr = q3 - q1
                            lower_bound = q1 - 3 * iqr
                            upper_bound = q3 + 3 * iqr

                            # Clip extreme outliers for better graph scaling
                            df_plot = df.copy()
                            df_plot[curve_to_plot] = df[curve_to_plot].clip(lower=lower_bound, upper=upper_bound)

                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                x=df_plot['Crank Angle'],
                                y=df_plot[curve_to_plot],
                                mode='lines',
                                name=curve_to_plot,
                                line=dict(color='#1f77b4', width=2)
                            ))

                            # Add horizontal lines for key features
                            fig.add_hline(
                                y=features['mean_amplitude'],
                                line_dash="dash",
                                line_color="green",
                                annotation_text=f"Mean: {features['mean_amplitude']:.1f} G",
                                annotation_position="right"
                            )
                            fig.add_hline(
                                y=features['median_amplitude'],
                                line_dash="dot",
                                line_color="orange",
                                annotation_text=f"Median: {features['median_amplitude']:.1f} G",
                                annotation_position="right"
                            )

                            # Calculate reasonable y-axis range
                            y_min = max(df_plot[curve_to_plot].min() - 2, 0)
                            y_max = df_plot[curve_to_plot].max() + 5

                            fig.update_layout(
                                title="AE Amplitude vs Crank Angle",
                                xaxis_title="Crank Angle (degrees)",
                                yaxis_title="Amplitude (G)",
                                yaxis_range=[y_min, y_max],
                                height=400,
                                hovermode='x unified',
                                showlegend=False
                            )

                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Could not visualize waveform")

                    with col_right:
                        st.subheader("🔢 Extracted Features")

                        # Features table with highlighting
                        feature_display = {
                            "Feature": [
                                "Mean Amplitude",
                                "Min Amplitude",
                                "Median Amplitude",
                                "Max Amplitude",
                                "Std Deviation",
                                "Amplitude Range",
                                "Crank Angle @ Max",
                                "Sample Count"
                            ],
                            "Value": [
                                f"{features['mean_amplitude']:.2f} G",
                                f"{features['min_amplitude']:.2f} G",
                                f"{features['median_amplitude']:.2f} G",
                                f"{features['max_amplitude']:.2f} G",
                                f"{features['std_amplitude']:.2f} G",
                                f"{features['amplitude_range']:.2f} G",
                                f"{features['crank_angle_at_max']:.1f}°",
                                f"{features['sample_count']}"
                            ],
                            "Importance": [
                                "⭐⭐⭐",  # Mean
                                "⭐⭐⭐⭐⭐",  # Min (most important)
                                "⭐⭐⭐⭐",  # Median
                                "⭐⭐",  # Max
                                "⭐",  # Std
                                "⭐",  # Range
                                "⭐",  # Crank angle
                                "⭐"   # Count
                            ]
                        }

                        st.dataframe(
                            pd.DataFrame(feature_display),
                            hide_index=True,
                            use_container_width=True
                        )

                        st.info("💡 **Key Indicators:** Min and Median amplitude are most important for leak detection (32.5% and 20.8% importance)")

                    # Technical details
                    with st.expander("🔬 Technical Details"):
                        st.markdown("""
                        **Model Information:**
                        - Algorithm: Random Forest Classifier
                        - Trees: 100
                        - Training samples: 53 valves (7 leak, 46 normal)
                        - Features: 8 statistical measures from AE waveforms
                        - Sensor type: AE/Ultrasonic (36-44 kHz narrow band)

                        **Performance Metrics (Test Set):**
                        - Accuracy: 81.8%
                        - Leak Recall: 100% (all leaks detected)
                        - Normal Precision: 80% (8/10 normals correct)
                        - False Alarms: 2/11 (acceptable for safety-critical application)

                        **Feature Importance:**
                        1. Min Amplitude (32.5%)
                        2. Median Amplitude (20.8%)
                        3. Mean Amplitude (19.5%)
                        4. Max Amplitude (13.0%)
                        5. Sample Count (5.3%)
                        """)

else:
    # No file uploaded yet
    st.info("👆 Upload a Curves XML file to begin analysis")

    # Sample download section
    st.markdown("---")
    st.subheader("📥 Need Sample Files?")
    st.markdown("""
    Don't have XML files to test? Sample files are included in the deployment package:
    - `sample_leak_1.xml` - Known leak valve
    - `sample_leak_2.xml` - Known leak valve
    - `sample_normal_1.xml` - Normal valve operation
    - `sample_normal_2.xml` - Normal valve operation

    Try uploading these to see how the system performs!
    """)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p><strong>Proof-of-Concept Demo</strong> | Preview</p>
    <p>AI-Powered Valve Leak Detection | Acoustic Emission Analysis</p>
    <p>Next Steps: Enhancements → 20 features + Ensemble models → 85-88% accuracy target</p>
</div>
""", unsafe_allow_html=True)

