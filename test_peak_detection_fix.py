"""
Test script to validate peak detection fix for system-wide leak detection
Tests on C402 Cylinder 3 and other known leaks/normals
"""

import sys
if sys.platform == 'win32':
    import codecs
    sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')

from xml_feature_extractor import extract_all_cylinders_features
import joblib
import pandas as pd

print("=" * 80)
print("PEAK DETECTION FIX VALIDATION TEST")
print("=" * 80)
print()

# Load model
print("Loading model...")
model = joblib.load('leak_detector_model.pkl')
scaler = joblib.load('feature_scaler.pkl')
print("Model loaded")
print()

# Test files
test_files = [
    {
        'name': 'C402 (Client-Confirmed Leak)',
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\C402_C_09_09_199812_02_53PM_Curves.xml',
        'expected_leak': 'Cylinder 3 Crank End Discharge (3CD1)'
    },
    {
        'name': '578-B (Training Data)',
        'path': r'C:\Users\Andrea\my-project\assets\xml-samples\578_B_09_25_20257_08_59AM_Curves.xml',
        'expected_leak': 'Cylinder 3 Head End Discharge (3HD1) - 13 training samples'
    }
]

results_summary = []

for test_file in test_files:
    print("=" * 80)
    print(f"TEST: {test_file['name']}")
    print(f"Expected Leak: {test_file['expected_leak']}")
    print("=" * 80)
    print()

    try:
        # Load XML
        with open(test_file['path'], 'r', encoding='utf-8') as f:
            xml_content = f.read()

        # Extract features
        all_valve_data = extract_all_cylinders_features(xml_content)

        if not all_valve_data:
            print(f"ERROR: Failed to extract valve data from {test_file['name']}")
            continue

        print(f"Extracted {len(all_valve_data)} valves")
        print()

        # Analyze each valve
        leak_detected = False
        max_leak_prob = 0
        max_leak_valve = None

        for valve in all_valve_data:
            # Prepare features
            feature_df = pd.DataFrame([valve['features']])
            feature_scaled = scaler.transform(feature_df)

            # Predict
            leak_probability = model.predict_proba(feature_scaled)[0][1] * 100
            prediction = "LEAK" if leak_probability >= 50.0 else "Normal"

            valve['leak_prob'] = leak_probability
            valve['status'] = prediction

            # Track maximum
            if leak_probability > max_leak_prob:
                max_leak_prob = leak_probability
                max_leak_valve = valve

            # Check for Cylinder 3
            if valve['cylinder_num'] == 3:
                emoji = "LEAK" if leak_probability >= 50.0 else "🟢 Normal"
                print(f"  Cylinder 3 {valve['valve_name']:25s} | {leak_probability:5.1f}% | {emoji}")
                print(f"    Features: mean={valve['features']['mean_amplitude']:.2f}G, "
                      f"max={valve['features']['max_amplitude']:.2f}G, "
                      f"count={valve['features']['sample_count']}")

                if leak_probability >= 50.0:
                    leak_detected = True

        print()
        print(f"Maximum leak probability: {max_leak_prob:.1f}% - {max_leak_valve['valve_name']} (Cylinder {max_leak_valve['cylinder_num']})")

        # Result
        if leak_detected and max_leak_prob >= 70.0:
            result = "PASS - Leak detected with proper confidence"
            status_emoji = ""
        elif leak_detected and max_leak_prob >= 50.0:
            result = "PARTIAL - Leak detected but low confidence"
            status_emoji = ""
        else:
            result = "FAIL - Leak not detected"
            status_emoji = ""

        print()
        print(f"Result: {status_emoji} {result}")

        results_summary.append({
            'file': test_file['name'],
            'max_leak_prob': max_leak_prob,
            'leak_detected': leak_detected,
            'result': result
        })

    except Exception as e:
        print(f"ERROR testing {test_file['name']}: {e}")
        import traceback
        traceback.print_exc()

    print()

# Summary
print("=" * 80)
print("TEST SUMMARY")
print("=" * 80)
print()

for result in results_summary:
    print(f"{result['file']}")
    print(f"  Max Leak Probability: {result['max_leak_prob']:.1f}%")
    print(f"  Leak Detected: {'Yes' if result['leak_detected'] else 'No'}")
    print(f"  Result: {result['result']}")
    print()

# Overall assessment
all_passed = all(r['max_leak_prob'] >= 70.0 for r in results_summary)

print("=" * 80)
if all_passed:
    print(" PEAK DETECTION FIX: SUCCESSFUL")
    print("All known leaks detected with proper confidence (70%+)")
else:
    print("PEAK DETECTION FIX: NEEDS ADJUSTMENT")
    print("Some leaks detected with low confidence (<70%)")
    print("May need to adjust peak detection parameters")
print("=" * 80)
