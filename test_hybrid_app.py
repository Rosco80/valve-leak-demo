"""
Quick test of hybrid detection system
Tests the updated app.py logic on known leak files
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from xml_feature_extractor import extract_all_cylinders_features
import joblib
import pandas as pd

print("=" * 100)
print("TESTING HYBRID ML + RULE-BASED DETECTION SYSTEM")
print("=" * 100)
print()

# Load model
print("Loading model...")
model = joblib.load('leak_detector_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
print("[OK] Model and scaler loaded")
print()

# Rule-based detection function (from app.py)
def rule_based_leak_detection(features):
    """
    Rule-based leak detection using pattern analysis
    Returns: (is_leak, confidence_score, criteria_met, score)
    """
    cont_ratio = features['continuity_ratio']
    spike_conc = features['spike_concentration']
    base_elev = features['baseline_elevation']
    iqr_score = features['iqr_score']

    criteria = {
        'High Continuity': cont_ratio > 0.55,
        'Low Spike Conc': spike_conc < 0.15,
        'High Baseline': base_elev > 0.40,
        'Low IQR': iqr_score < 0.20
    }

    score = sum(criteria.values())

    if score >= 3:
        is_leak = True
        confidence = 0.75 + (score - 3) * 0.10
    elif score == 2:
        is_leak = False
        confidence = 0.50
    else:
        is_leak = False
        confidence = 0.25 + (2 - score) * 0.10

    return is_leak, confidence, criteria, score

# Test files (known leaks)
test_files = [
    {
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\C402_C_09_09_199812_02_53PM_Curves.xml',
        'name': 'C402 Sep 9 1998',
        'known_leak': 'Cyl 3 CD'
    },
    {
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\578_B_09_25_20257_08_59AM_Curves.xml',
        'name': '578-B Sep 25 2002',
        'known_leak': 'Cyl 3 Multiple'
    }
]

for file_info in test_files:
    print("=" * 100)
    print(f"FILE: {file_info['name']} ⭐ Known Leak: {file_info['known_leak']}")
    print("=" * 100)

    try:
        # Load XML
        with open(file_info['path'], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Extract features
        all_cylinder_data = extract_all_cylinders_features(xml_content)

        if not all_cylinder_data:
            print("  ERROR: Failed to extract features")
            continue

        # Make predictions using HYBRID approach (same logic as app.py)
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

            # HYBRID DECISION
            if ml_detected and rule_leak:
                prediction = 1
                confidence = max(probabilities[1], rule_conf)
                confidence_level = "High Confidence"
            elif ml_detected or rule_leak:
                prediction = 1
                confidence = (probabilities[1] + rule_conf) / 2
                confidence_level = "Moderate - Inspection Recommended"
            else:
                prediction = 0
                confidence = probabilities[0]
                confidence_level = "Normal"

            # Store results
            valve_data['prediction'] = prediction
            valve_data['ml_probability'] = ml_leak_probability
            valve_data['ml_detected'] = ml_detected
            valve_data['rule_score'] = rule_score
            valve_data['rule_detected'] = rule_leak
            valve_data['confidence_level'] = confidence_level

        # Show Cylinder 3 results
        cyl3_valves = [v for v in all_cylinder_data if v['cylinder_num'] == 3]

        if cyl3_valves:
            print()
            print("  CYLINDER 3 HYBRID RESULTS:")
            print("  " + "-" * 96)
            print(f"  {'Valve':<8} {'ML Prob':<10} {'ML':<8} {'Rule':<8} {'Rule':<8} {'Final':<15} {'Confidence':<20}")
            print("  " + "-" * 96)

            for v in cyl3_valves:
                ml_icon = "✅" if v['ml_detected'] else "❌"
                rule_icon = "✅" if v['rule_detected'] else "❌"
                final = "⚠️ LEAK" if v['prediction'] == 1 else "✅ Normal"

                print(f"  {v['valve_position']:<8} "
                      f"{v['ml_probability']:>8.1f}% "
                      f"{ml_icon:<8} "
                      f"{v['rule_score']}/4 "
                      f"{rule_icon:<8} "
                      f"{final:<15} "
                      f"{v['confidence_level']:<20}")

            print()

        print()

    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()
        print()

print("=" * 100)
print("HYBRID SYSTEM TEST COMPLETE")
print("=" * 100)
print()
print("✅ If both C402 and 578-B show leaks detected in Cyl 3, hybrid system is working!")
