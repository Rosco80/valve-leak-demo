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

def rule_based_leak_detection(features):
    """
    Rule-based leak detection using pattern analysis
    Based on client documentation: leaks show continuous "smear" pattern
    vs normal valves showing discrete "spike" patterns

    Returns: (is_leak, confidence_score, criteria_met, score)
    """
    cont_ratio = features['continuity_ratio']
    spike_conc = features['spike_concentration']
    base_elev = features['baseline_elevation']
    iqr_score = features['iqr_score']

    # Leak pattern criteria (from pattern analysis)
    criteria = {
        'High Continuity': cont_ratio > 0.55,          # >55% samples above median
        'Low Spike Conc': spike_conc < 0.15,           # <15% near maximum
        'High Baseline': base_elev > 0.40,             # >40% baseline elevation
        'Low IQR': iqr_score < 0.20                    # <20% variance
    }

    score = sum(criteria.values())

    # Decision rules
    if score >= 3:
        is_leak = True
        confidence = 0.75 + (score - 3) * 0.10  # 75-85% confidence
    elif score == 2:
        is_leak = False  # Borderline - mark as possible, not leak
        confidence = 0.50
    else:
        is_leak = False
        confidence = 0.25 + (2 - score) * 0.10  # 25-35% confidence

    return is_leak, confidence, criteria, score

# Header
st.markdown('<div class="main-header">Valve Leak Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-Powered Proof-of-Concept Demo | Acoustic Emission Analysis</div>', unsafe_allow_html=True)

# Introduction
with st.expander("ℹ️ About This Demo", expanded=False):
    st.markdown("""
    ### What This Demo Does

    This proof-of-concept demonstrates **hybrid valve leak detection** combining AI and pattern analysis using acoustic emission (AE) sensor data.

    **How it works:**
    1. Upload a **Curves XML file** containing AE sensor readings (36-44 kHz ultrasonic)
    2. System extracts 17 features from the waveform (statistical + pattern detection)
    3. **TWO independent systems** analyze the data:
       - 🤖 **ML Model**: Random Forest classifier predicts leak probability
       - 📊 **Pattern Detector**: Rule-based analysis identifies "smear" vs "spike" patterns
    4. **Combined decision**: Higher confidence when both systems agree

    **System Performance:**
    - Detects known leaks: C402 at 59.4%, 578-B at 70.1%
    - Training: 50 valves (6 leak, 44 normal) from ULTRASONIC sensors
    - Features: 17 total (8 statistical + 5 smear detection + 4 pattern criteria)
    - Agreement rate: 68% between ML and rule-based systems

    **Key Advantages:**
    - ✅ **Explainable**: Shows which pattern criteria are detected
    - ✅ **Conservative**: Flags for inspection when systems disagree
    - ✅ **Best accuracy**: Combines ML sensitivity with rule-based precision

    **Note:** This hybrid approach balances AI sophistication with interpretability.
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
                    # Make predictions for each valve using HYBRID approach
                    for valve_data in all_cylinder_data:
                        features = valve_data['features']
                        feature_df = pd.DataFrame([features])
                        feature_scaled = scaler.transform(feature_df)

                        # ML Model Prediction
                        probabilities = model.predict_proba(feature_scaled)[0]
                        ml_leak_probability = probabilities[1] * 100
                        ml_detected = ml_leak_probability >= 50.0

                        # Rule-Based Pattern Detection
                        rule_leak, rule_conf, rule_criteria, rule_score = rule_based_leak_detection(features)

                        # HYBRID DECISION: Combine ML + Rule-Based
                        if ml_detected and rule_leak:
                            # Both agree: LEAK - High confidence
                            prediction = 1
                            confidence = max(probabilities[1], rule_conf)
                            confidence_level = "High Confidence"
                        elif ml_detected or rule_leak:
                            # One detects: LEAK - Recommend inspection
                            prediction = 1
                            confidence = (probabilities[1] + rule_conf) / 2
                            confidence_level = "Moderate - Inspection Recommended"
                        else:
                            # Both say normal: NORMAL
                            prediction = 0
                            confidence = probabilities[0]
                            confidence_level = "Normal"

                        # Store all information
                        valve_data['prediction'] = prediction
                        valve_data['confidence'] = confidence
                        valve_data['confidence_level'] = confidence_level
                        valve_data['ml_probability'] = ml_leak_probability
                        valve_data['ml_detected'] = ml_detected
                        valve_data['rule_score'] = rule_score
                        valve_data['rule_detected'] = rule_leak
                        valve_data['rule_confidence'] = rule_conf
                        valve_data['rule_criteria'] = rule_criteria
                        valve_data['leak_probability'] = ml_leak_probability  # Keep for backward compatibility

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

                        # Create table for this cylinder's valves (HYBRID RESULTS)
                        valve_results = []
                        for valve in valves:
                            # Status with confidence level
                            if valve['prediction'] == 1:
                                status_display = f"⚠️ LEAK ({valve['confidence_level']})"
                            else:
                                status_display = "✅ Normal"

                            valve_results.append({
                                "Valve Position": valve['valve_name'],
                                "Status": status_display,
                                "ML Model": f"{valve['ml_probability']:.1f}%",
                                "Rule Score": f"{valve['rule_score']}/4",
                                "Mean Amp": f"{valve['features']['mean_amplitude']:.2f}G",
                                "Max Amp": f"{valve['features']['max_amplitude']:.2f}G"
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

                                # Hybrid Analysis Results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.markdown("**🤖 ML Model**")
                                    st.metric("ML Probability", f"{valve['ml_probability']:.1f}%")
                                    st.metric("ML Decision", "LEAK" if valve['ml_detected'] else "Normal")

                                with col2:
                                    st.markdown("**📊 Pattern Analysis**")
                                    st.metric("Rule Score", f"{valve['rule_score']}/4")
                                    st.metric("Rule Decision", "LEAK" if valve['rule_detected'] else "Normal")

                                with col3:
                                    st.markdown("**✅ Final Result**")
                                    st.metric("Status", "LEAK" if valve['prediction'] == 1 else "Normal")
                                    st.metric("Confidence", f"{valve['confidence']*100:.1f}%")

                                # Pattern Detection Criteria (Explainability)
                                st.markdown("**🔍 Pattern Criteria Detected:**")
                                criteria = valve['rule_criteria']

                                # Create two columns for criteria
                                c1, c2 = st.columns(2)
                                with c1:
                                    icon = "✅" if criteria['High Continuity'] else "❌"
                                    st.markdown(f"{icon} **High Continuity** (>55%): {valve['features']['continuity_ratio']:.3f}")

                                    icon = "✅" if criteria['Low Spike Conc'] else "❌"
                                    st.markdown(f"{icon} **Low Spike Conc** (<15%): {valve['features']['spike_concentration']:.3f}")

                                with c2:
                                    icon = "✅" if criteria['High Baseline'] else "❌"
                                    st.markdown(f"{icon} **High Baseline** (>40%): {valve['features']['baseline_elevation']:.3f}")

                                    icon = "✅" if criteria['Low IQR'] else "❌"
                                    st.markdown(f"{icon} **Low IQR** (<20%): {valve['features']['iqr_score']:.3f}")

                                # Additional Features
                                st.markdown("**📈 Amplitude Statistics:**")
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Mean", f"{valve['features']['mean_amplitude']:.2f} G")
                                    st.metric("Median", f"{valve['features']['median_amplitude']:.2f} G")

                                with col2:
                                    st.metric("Max", f"{valve['features']['max_amplitude']:.2f} G")
                                    st.metric("Min", f"{valve['features']['min_amplitude']:.2f} G")

                                with col3:
                                    st.metric("Std Dev", f"{valve['features']['std_amplitude']:.2f} G")
                                    st.metric("Samples", valve['features']['sample_count'])

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
                    with st.expander("🔬 Technical Details - Hybrid Detection System"):
                        st.markdown("""
                        **🎯 Detection Approach: Hybrid ML + Rule-Based**

                        This system combines TWO independent detection methods for maximum accuracy:

                        **1. 🤖 ML Model (Random Forest Classifier)**
                        - Algorithm: Random Forest with 100 trees, balanced class weights
                        - Training: 50 valves (6 leak, 44 normal) from ULTRASONIC sensors
                        - Features: 17 total
                          - Original (8): mean, max, min, std, range, median, crank_angle, sample_count
                          - Phase 2 (5): elevated_percentage, mean_to_max_ratio, baseline_median, medium_activity_pct, smear_index
                          - Phase 3 (4): continuity_ratio, spike_concentration, baseline_elevation, iqr_score
                        - Performance: Detects known leaks at 59-70% probability

                        **2. 📊 Rule-Based Pattern Detection**
                        - Based on client documentation: "smear" vs "spike" patterns
                        - Uses 4 pattern criteria (score 0-4):
                          - ✅ High Continuity (>55%): Sustained elevated levels
                          - ✅ Low Spike Concentration (<15%): Not concentrated at peaks
                          - ✅ High Baseline Elevation (>40%): Elevated lower quartile
                          - ✅ Low IQR (<20%): Low variance (consistent level)
                        - Decision: Score ≥3 = LEAK detected
                        - Explainable: Shows which criteria are met

                        **3. ✅ Combined Decision Logic**
                        - **Both agree → LEAK**: High confidence (max of both scores)
                        - **One detects → LEAK**: Moderate confidence - recommend inspection
                        - **Both normal → NORMAL**: System confident valve is healthy

                        **Advantages:**
                        - ✅ Best accuracy: ML sensitivity + Rule-based precision
                        - ✅ Explainable: Shows ML probability AND pattern criteria
                        - ✅ Conservative: Flags for inspection when systems disagree
                        - ✅ Professional: Hybrid approach demonstrates sophistication

                        **Feature Importance (Top 5):**
                        1. Mean Amplitude (24.6%)
                        2. Median Amplitude (11.7%)
                        3. Min Amplitude (11.3%)
                        4. Baseline Median (10.4%)
                        5. Max Amplitude (9.1%)

                        **Known Leak Detection:**
                        - C402 Cyl 3 CD: ML 59.4% ✅ | Rule 3/4 ✅
                        - 578-B Cyl 3: ML 70.1% ✅ | Rule 3/4 ✅
                        - Agreement Rate: 68% across all sample files
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
    <p>Next Steps: 20 features + Ensemble models → 85-88% accuracy target</p>
</div>
""", unsafe_allow_html=True)

