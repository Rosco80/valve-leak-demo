"""
XML Feature Extractor for Leak Detection Demo
Parses Curves XML files and extracts 8 features for model inference
"""

import xml.etree.ElementTree as ET
import pandas as pd
import re
from typing import Optional, Dict, Any

# XML namespace for Microsoft Office Spreadsheet format
XML_NS = {'ss': 'urn:schemas-microsoft-com:office:spreadsheet'}


def parse_curves_xml(xml_content: str) -> Optional[pd.DataFrame]:
    """
    Parse Curves XML file and return DataFrame with amplitude data.

    Args:
        xml_content: XML content string

    Returns:
        DataFrame with 'Crank Angle' and amplitude columns, or None if error
    """
    try:
        root = ET.fromstring(xml_content)
        ws = next(
            (ws for ws in root.findall('.//ss:Worksheet', XML_NS)
             if ws.attrib.get('{urn:schemas-microsoft-com:office:spreadsheet}Name') == 'Curves'),
            None
        )
        if ws is None:
            return None

        table = ws.find('.//ss:Table', XML_NS)
        if table is None:
            return None

        rows = table.findall('ss:Row', XML_NS)

        # Extract headers (row 1, index 1 in zero-indexed list)
        header_cells = rows[1].findall('ss:Cell', XML_NS)
        data_elements = [c.find('ss:Data', XML_NS) for c in header_cells]
        raw_headers = [(elem.text if elem is not None and elem.text else '') for elem in data_elements]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:] if name]

        # Extract data (skip rows 6-11 which contain summary rows like "Overall", "Median Period", etc.)
        # Start from row 12 to get actual crank angle waveform data
        raw_data = []
        for r in rows[12:]:
            row_cells = r.findall('ss:Cell', XML_NS)
            row_data = []
            for cell in row_cells:
                data_elem = cell.find('ss:Data', XML_NS)
                row_data.append(data_elem.text if data_elem is not None else None)
            raw_data.append(row_data)

        # Filter to only include rows with numeric crank angles (actual waveform data)
        data = []
        for row in raw_data:
            if row and row[0]:  # Check if row has data
                try:
                    float(row[0])  # Verify first column (crank angle) is numeric
                    data.append(row)
                except (ValueError, TypeError):
                    continue  # Skip non-numeric rows (metadata/summary rows)

        if not data:
            return None

        num_data_columns = len(data[0])
        actual_columns = full_header_list[:num_data_columns]

        # Create DataFrame and clean
        df = pd.DataFrame(data, columns=actual_columns)
        df = df.apply(pd.to_numeric, errors='coerce')
        df = df.dropna(how='all').dropna(axis=1, how='all')
        df.sort_values('Crank Angle', inplace=True)

        return df

    except Exception as e:
        print(f"Failed to parse curves XML: {e}")
        return None


def extract_features_from_xml(xml_content: str, curve_name: Optional[str] = None) -> Optional[Dict[str, float]]:
    """
    Extract 8 features from Curves XML file for leak detection.

    Args:
        xml_content: XML content string
        curve_name: Optional specific curve/column name to analyze.
                   If None, uses first AE/Ultrasonic curve found.

    Returns:
        Dictionary with 8 features, or None if error
    """
    try:
        # Parse XML
        df = parse_curves_xml(xml_content)
        if df is None or len(df) == 0:
            return None

        # Find AE/Ultrasonic curve column if not specified
        if curve_name is None:
            # Priority 1: Look for exact training curve type (ULTRASONIC G 36KHZ - 44KHZ)
            ae_columns = [col for col in df.columns
                         if col != 'Crank Angle' and
                         'ULTRASONIC' in col.upper() and '36KHZ' in col.upper() and '44KHZ' in col.upper()]

            # Priority 2: Any 36KHZ ultrasonic curve
            if not ae_columns:
                ae_columns = [col for col in df.columns
                             if col != 'Crank Angle' and
                             'ULTRASONIC' in col.upper() and '36KHZ' in col.upper()]

            # Priority 3: Any ultrasonic/AE curve
            if not ae_columns:
                ae_columns = [col for col in df.columns
                             if col != 'Crank Angle' and
                             ('ULTRASONIC' in col.upper() or 'AE' in col.upper() or 'KHZ' in col.upper())]

            # Priority 4: Fallback to first non-crank-angle column
            if not ae_columns:
                ae_columns = [col for col in df.columns if col != 'Crank Angle']

            if not ae_columns:
                return None

            curve_name = ae_columns[0]

        # Extract amplitude values
        amplitude_values_full = df[curve_name].dropna()

        if len(amplitude_values_full) == 0:
            return None

        # SYSTEM-WIDE FIX: Peak Detection Alignment
        # Root cause: Training data used peak-detected samples (2-13 peaks per valve)
        # but inference was using full waveform (355 points). This creates feature mismatch.
        #
        # Solution: Apply peak detection to match training methodology
        # This restores system functionality across ALL files and valves

        try:
            from scipy.signal import find_peaks

            # Method 1: Use scipy peak detection (preferred - more robust)
            # Detect peaks with minimum height at 90th percentile
            threshold = amplitude_values_full.quantile(0.90)

            # Find peaks with:
            # - height: at 90th percentile
            # - distance: minimum 5 samples between peaks (avoids noise)
            peak_indices, peak_properties = find_peaks(
                amplitude_values_full.values,
                height=threshold,
                distance=5
            )

            if len(peak_indices) >= 2:
                # Use detected peaks (matches training data format)
                amplitude_values = amplitude_values_full.iloc[peak_indices]
            else:
                # Fallback: Use threshold-based selection if too few peaks found
                amplitude_values = amplitude_values_full[amplitude_values_full >= threshold]

                # Safety: Ensure minimum samples
                if len(amplitude_values) < 2:
                    amplitude_values = amplitude_values_full.nlargest(max(3, int(len(amplitude_values_full) * 0.05)))

        except ImportError:
            # Fallback if scipy not available: Use threshold-based method
            # (Matches original training data creation method)
            threshold_percentile = amplitude_values_full.quantile(0.90)
            threshold_adaptive = amplitude_values_full.mean() + 2 * amplitude_values_full.std()
            threshold = max(threshold_percentile, threshold_adaptive, 3.0)

            amplitude_values = amplitude_values_full[amplitude_values_full >= threshold]

            # Ensure minimum samples
            if len(amplitude_values) < 2:
                amplitude_values = amplitude_values_full.nlargest(max(3, int(len(amplitude_values_full) * 0.05)))

        # Calculate 13 features for enhanced leak detection
        mean_val = amplitude_values.mean()
        max_val = amplitude_values.max()
        min_val = amplitude_values.min()
        std_val = amplitude_values.std()
        median_val = amplitude_values.median()

        # Additional leak-specific features
        # 1. Elevated percentage: % of points above median (leak = sustained elevation)
        elevated_count = (amplitude_values > median_val).sum()
        elevated_percentage = (elevated_count / len(amplitude_values)) * 100

        # 2. Mean-to-max ratio: leak = high ratio (smear), normal = low ratio (spikes)
        mean_to_max_ratio = mean_val / max_val if max_val > 0 else 0

        # 3. Baseline median: Using lower 50% of values to establish baseline
        lower_half = amplitude_values[amplitude_values <= median_val]
        baseline_median = lower_half.median() if len(lower_half) > 0 else median_val

        # 4. Medium activity percentage: % of points in 50-90th percentile range
        p50 = amplitude_values.quantile(0.50)
        p90 = amplitude_values.quantile(0.90)
        medium_activity = ((amplitude_values >= p50) & (amplitude_values <= p90)).sum()
        medium_activity_pct = (medium_activity / len(amplitude_values)) * 100

        # 5. Smear index: std deviation of amplitude values (leak = low variance smear)
        smear_index = std_val / mean_val if mean_val > 0 else 0

        # PHASE 3: Pattern Detection Features (Smear vs Spike)
        # Based on client documentation showing continuous smear (leak) vs discrete spikes (normal)

        # 6. Continuity Ratio: What % of samples are above median
        # Smear pattern (leak): 60-80% above median (continuous elevation)
        # Spike pattern (normal): 10-30% above median (discrete peaks)
        above_median_count = (amplitude_values > median_val).sum()
        continuity_ratio = above_median_count / len(amplitude_values) if len(amplitude_values) > 0 else 0

        # 7. Spike Concentration Index: What % of samples are near maximum (>80% of max)
        # Spike pattern (normal): HIGH concentration (5-15% near max, rest low)
        # Smear pattern (leak): LOW concentration (<5% near max, distributed)
        near_max_threshold = max_val * 0.8
        near_max_count = (amplitude_values >= near_max_threshold).sum()
        spike_concentration = near_max_count / len(amplitude_values) if len(amplitude_values) > 0 else 0

        # 8. Baseline Elevation: Lower quartile (25th percentile) relative to max
        # Smear pattern (leak): HIGH ratio (0.4-0.6) - even low values are elevated
        # Spike pattern (normal): LOW ratio (0.0-0.2) - low values near zero
        q25 = amplitude_values.quantile(0.25)
        baseline_elevation = q25 / max_val if max_val > 0 else 0

        # 9. IQR Score: Inter-quartile range relative to max amplitude
        # Smear pattern (leak): SMALL IQR (values clustered around continuous level)
        # Spike pattern (normal): LARGE IQR (wide variation from baseline to peaks)
        q75 = amplitude_values.quantile(0.75)
        iqr = q75 - q25
        iqr_score = iqr / max_val if max_val > 0 else 0

        features = {
            # Original 8 features
            'mean_amplitude': float(mean_val),
            'max_amplitude': float(max_val),
            'min_amplitude': float(min_val),
            'std_amplitude': float(std_val),  # type: ignore
            'amplitude_range': float(max_val - min_val),
            'median_amplitude': float(median_val),
            'crank_angle_at_max': float(df.loc[amplitude_values.idxmax(), 'Crank Angle']),  # type: ignore[arg-type]
            'sample_count': int(len(amplitude_values)),

            # Phase 2: 5 leak-detection features
            'elevated_percentage': float(elevated_percentage),
            'mean_to_max_ratio': float(mean_to_max_ratio),
            'baseline_median': float(baseline_median),
            'medium_activity_pct': float(medium_activity_pct),
            'smear_index': float(smear_index),

            # Phase 3: 4 pattern detection features (smear vs spike)
            'continuity_ratio': float(continuity_ratio),
            'spike_concentration': float(spike_concentration),
            'baseline_elevation': float(baseline_elevation),
            'iqr_score': float(iqr_score)
        }

        return features

    except Exception as e:
        print(f"Failed to extract features: {e}")
        return None


def extract_all_cylinders_features(xml_content: str):
    """
    Extract features from ALL cylinders/valves in an XML file.

    Args:
        xml_content: XML content string

    Returns:
        List of dictionaries, each containing:
        - cylinder_num: Cylinder number (1-N)
        - valve_position: Valve position code (e.g., 'CS1', 'CD1', 'HS1', 'HD1')
        - valve_name: Full valve description
        - column_name: Original XML column name
        - features: Dictionary of 8 extracted features
        Returns None if parsing fails
    """
    try:
        # Parse XML
        df = parse_curves_xml(xml_content)
        if df is None or len(df) == 0:
            return None

        # Find all ULTRASONIC columns (36KHZ - 44KHZ preferred)
        ultrasonic_cols = [col for col in df.columns
                          if col != 'Crank Angle' and
                          'ULTRASONIC' in col.upper() and '36KHZ' in col.upper() and '44KHZ' in col.upper()]

        # Fallback to any ultrasonic/AE curve if specific frequency not found
        if not ultrasonic_cols:
            ultrasonic_cols = [col for col in df.columns
                              if col != 'Crank Angle' and
                              ('ULTRASONIC' in col.upper() or 'AE' in col.upper())]

        if not ultrasonic_cols:
            return None

        results = []

        for col_name in ultrasonic_cols:
            # Extract cylinder number and valve position from column name
            # Format: "C402 - C.3CS1.ULTRASONIC G 36KHZ - 44KHZ (NARROW BAND).3CS1"
            # We need to extract: cylinder=3, position=CS1

            # Try to find cylinder number pattern (C.1, C.2, C.3, etc.)
            import re
            cyl_match = re.search(r'C\.(\d+)([A-Z]{2,3}\d*)', col_name)

            if cyl_match:
                cylinder_num = int(cyl_match.group(1))
                valve_pos = cyl_match.group(2)
            else:
                # Fallback: try to find any digit pattern
                digit_match = re.search(r'\.(\d+)', col_name)
                if digit_match:
                    cylinder_num = int(digit_match.group(1))
                    valve_pos = "Unknown"
                else:
                    continue  # Skip if can't extract cylinder number

            # Decode valve position
            valve_position_map = {
                'CS1': 'Crank End Suction',
                'CD1': 'Crank End Discharge',
                'HS1': 'Head End Suction',
                'HD1': 'Head End Discharge',
                'CS': 'Crank End Suction',
                'CD': 'Crank End Discharge',
                'HS': 'Head End Suction',
                'HD': 'Head End Discharge'
            }

            valve_name = valve_position_map.get(valve_pos, valve_pos)

            # Extract features for this specific column
            features = extract_features_from_xml(xml_content, curve_name=col_name)

            if features is not None:
                results.append({
                    'cylinder_num': cylinder_num,
                    'valve_position': valve_pos,
                    'valve_name': valve_name,
                    'column_name': col_name,
                    'features': features
                })

        # Sort by cylinder number and valve position
        results.sort(key=lambda x: (x['cylinder_num'], x['valve_position']))

        return results

    except Exception as e:
        print(f"Failed to extract all cylinder features: {e}")
        return None


def get_curve_info(xml_content: str) -> Dict[str, Any]:
    """
    Extract metadata from Curves XML file.

    Args:
        xml_content: XML content string

    Returns:
        Dictionary with curve metadata
    """
    try:
        df = parse_curves_xml(xml_content)
        if df is None:
            return {'error': 'Failed to parse XML'}

        # Get curve names (exclude Crank Angle)
        curves = [col for col in df.columns if col != 'Crank Angle']

        # Find AE curves
        ae_curves = [col for col in curves
                    if 'ULTRASONIC' in col.upper() or 'AE' in col.upper() or 'KHZ' in col.upper()]

        return {
            'total_curves': len(curves),
            'curve_names': curves,
            'ae_curves': ae_curves,
            'data_points': len(df),
            'crank_angle_range': f"{df['Crank Angle'].min():.0f}-{df['Crank Angle'].max():.0f}°"
        }

    except Exception as e:
        return {'error': str(e)}
