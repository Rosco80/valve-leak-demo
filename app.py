"""
Valve Leak Detection - Cloud Demo
Streamlit app for Monday proof-of-concept demonstration
"""

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from xml_feature_extractor import extract_features_from_xml, parse_curves_xml, get_curve_info, extract_all_cylinders_features
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
st.markdown('<div class="main-header">🔧 Valve Leak Detection System</div>', unsafe_allow_html=True)
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
    if st.button("🔍 Analyze All Cylinders", type="primary", use_container_width=True):
        with st.spinner("Extracting features from all cylinders and running AI model..."):
            # Extract features from all cylinders
            all_cylinder_data = extract_all_cylinders_features(xml_content)

            if all_cylinder_data is None or len(all_cylinder_data) == 0:
                st.error("❌ Failed to extract features from XML file. Please check file format.")
            else:
                # Load model
                model, scaler = load_model()

                if model is None:
                    st.error("❌ Failed to load model. Please contact support.")
                else:
                    # Make predictions for each valve
                    for valve_data in all_cylinder_data:
                        feature_df = pd.DataFrame([valve_data['features']])
                        feature_scaled = scaler.transform(feature_df)

                        probabilities = model.predict_proba(feature_scaled)[0]
                        leak_probability = probabilities[1] * 100

                        # Use standard 50% threshold (proper peak detection fix restored system functionality)
                        # Previous 40% threshold was a band-aid for feature extraction mismatch
                        prediction = 1 if leak_probability >= 50.0 else 0

                        valve_data['prediction'] = prediction
                        valve_data['confidence'] = probabilities[prediction]
                        valve_data['leak_probability'] = leak_probability

                    # Group results by cylinder
                    cylinders = {}
                    for valve_data in all_cylinder_data:
                        cyl_num = valve_data['cylinder_num']
                        if cyl_num not in cylinders:
                            cylinders[cyl_num] = []
                        cylinders[cyl_num].append(valve_data)

                    # Calculate cylinder-level status (if ANY valve leaks, cylinder has leak)
                    cylinder_status = {}
                    for cyl_num, valves in cylinders.items():
                        max_leak_prob = max(v['leak_probability'] for v in valves)
                        has_leak = any(v['prediction'] == 1 for v in valves)
                        cylinder_status[cyl_num] = {
                            'has_leak': has_leak,
                            'max_leak_prob': max_leak_prob,
                            'leak_count': sum(1 for v in valves if v['prediction'] == 1)
                        }

                    # Display overall summary
                    st.markdown("## 🎯 Multi-Cylinder Analysis Results")

                    leaking_cylinders = [cyl for cyl, status in cylinder_status.items() if status['has_leak']]

                    if leaking_cylinders:
                        st.markdown(
                            f'<div class="result-box leak-detected">⚠️ LEAKS DETECTED IN {len(leaking_cylinders)} CYLINDER(S)</div>',
                            unsafe_allow_html=True
                        )
                        st.error(f"**Leaking Cylinders:** {', '.join([f'Cylinder {c}' for c in sorted(leaking_cylinders)])}")
                        st.warning("**Recommendation:** Schedule immediate maintenance inspection for affected cylinders.")
                    else:
                        st.markdown(
                            f'<div class="result-box normal-detected">✅ ALL CYLINDERS NORMAL</div>',
                            unsafe_allow_html=True
                        )
                        st.success("**Status:** All valves operating within normal parameters.")

                    st.markdown("---")

                    # Display per-cylinder results
                    st.subheader("📊 Cylinder-by-Cylinder Breakdown")

                    for cyl_num in sorted(cylinders.keys()):
                        valves = cylinders[cyl_num]
                        status = cylinder_status[cyl_num]

                        # Cylinder header with color coding
                        if status['has_leak']:
                            st.markdown(f"### 🔴 Cylinder {cyl_num} - LEAK DETECTED ({status['leak_count']} valve(s))")
                        else:
                            st.markdown(f"### 🟢 Cylinder {cyl_num} - Normal")

                        # Create table for this cylinder's valves
                        valve_results = []
                        for valve in valves:
                            status_emoji = "⚠️ LEAK" if valve['prediction'] == 1 else "✅ Normal"
                            valve_results.append({
                                "Valve Position": valve['valve_name'],
                                "Status": status_emoji,
                                "Leak Probability": f"{valve['leak_probability']:.1f}%",
                                "Mean Amplitude": f"{valve['features']['mean_amplitude']:.2f} G",
                                "Max Amplitude": f"{valve['features']['max_amplitude']:.2f} G"
                            })

                        df_results = pd.DataFrame(valve_results)

                        # Color code the dataframe
                        def highlight_leaks(row):
                            if "LEAK" in row['Status']:
                                return ['background-color: #ffebee'] * len(row)
                            return [''] * len(row)

                        st.dataframe(
                            df_results.style.apply(highlight_leaks, axis=1),
                            hide_index=True,
                            use_container_width=True
                        )

                        # Add detailed view in expander
                        with st.expander(f"🔬 Detailed Analysis - Cylinder {cyl_num}"):
                            for valve in valves:
                                st.markdown(f"**{valve['valve_name']}** ({valve['valve_position']})")

                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Leak Probability", f"{valve['leak_probability']:.1f}%")
                                    st.metric("Mean Amplitude", f"{valve['features']['mean_amplitude']:.2f} G")
                                    st.metric("Median Amplitude", f"{valve['features']['median_amplitude']:.2f} G")
                                    st.metric("Min Amplitude", f"{valve['features']['min_amplitude']:.2f} G")

                                with col2:
                                    st.metric("Confidence", f"{valve['confidence']:.1f}%")
                                    st.metric("Max Amplitude", f"{valve['features']['max_amplitude']:.2f} G")
                                    st.metric("Std Deviation", f"{valve['features']['std_amplitude']:.2f} G")
                                    st.metric("Sample Count", valve['features']['sample_count'])

                                st.markdown("---")

                        st.markdown("")  # Spacing

                    st.markdown("---")

                    # Summary Visualization
                    st.subheader("📊 Leak Probability Summary - All Cylinders")

                    # Create bar chart showing leak probability for each cylinder
                    cyl_summary = []
                    for cyl_num in sorted(cylinders.keys()):
                        status = cylinder_status[cyl_num]
                        cyl_summary.append({
                            'Cylinder': f"Cyl {cyl_num}",
                            'Max Leak Probability': status['max_leak_prob'],
                            'Status': 'LEAK' if status['has_leak'] else 'Normal'
                        })

                    df_summary = pd.DataFrame(cyl_summary)

                    fig = go.Figure()

                    # Color bars based on leak status
                    colors = ['#c62828' if row['Status'] == 'LEAK' else '#2e7d32'
                             for _, row in df_summary.iterrows()]

                    fig.add_trace(go.Bar(
                        x=df_summary['Cylinder'],
                        y=df_summary['Max Leak Probability'],
                        marker_color=colors,
                        text=df_summary['Max Leak Probability'].apply(lambda x: f"{x:.1f}%"),
                        textposition='outside'
                    ))

                    # Add threshold line at 50%
                    fig.add_hline(
                        y=50,
                        line_dash="dash",
                        line_color="orange",
                        annotation_text="50% Threshold",
                        annotation_position="right"
                    )

                    fig.update_layout(
                        title="Maximum Leak Probability by Cylinder",
                        xaxis_title="Cylinder",
                        yaxis_title="Leak Probability (%)",
                        yaxis_range=[0, 105],
                        height=400,
                        showlegend=False
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.info("💡 **Note:** Each cylinder bar shows the HIGHEST leak probability among its 4 valves. Expand individual cylinders above for valve-level details.")

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
    <p><strong>Monday Proof-of-Concept Demo</strong> | Free Preview (Not part of Week 1-4 pilot)</p>
    <p>AI-Powered Valve Leak Detection | Acoustic Emission Analysis</p>
    <p>Next Steps: Week 2-4 Pilot → 20 features + Ensemble models → 85-88% accuracy target</p>
</div>
""", unsafe_allow_html=True)
