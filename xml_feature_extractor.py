"""
XML Feature Extractor for Leak Detection Demo
Parses Curves XML files and extracts 8 features for model inference
"""

import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
import re
from typing import Optional, Dict

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
        rows = table.findall('ss:Row', XML_NS)

        # Extract headers (row 1, index 1 in zero-indexed list)
        header_cells = rows[1].findall('ss:Cell', XML_NS)
        raw_headers = [c.find('ss:Data', XML_NS).text or '' for c in header_cells]
        full_header_list = ["Crank Angle"] + [re.sub(r'\s+', ' ', name.strip()) for name in raw_headers[1:]]

        # Extract data (skip rows 6-11 which contain summary rows like "Overall", "Median Period", etc.)
        # Start from row 12 to get actual crank angle waveform data
        raw_data = [[cell.find('ss:Data', XML_NS).text for cell in r.findall('ss:Cell', XML_NS)] for r in rows[12:]]

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

        # EVENT-LEVEL FILTERING: Extract only significant amplitude events to match CSV training data format
        # The CSV training data contains sparse event points (peaks/significant events), not full waveforms

        # Strategy: Use 90th percentile to extract only the strongest events
        # This mimics the CSV format which contains only significant amplitude spikes
        threshold_percentile = amplitude_values_full.quantile(0.90)

        # Also try adaptive threshold: mean + 2*std (statistical outliers)
        threshold_adaptive = amplitude_values_full.mean() + 2 * amplitude_values_full.std()

        # Use the higher of the two thresholds to be more selective
        threshold = max(threshold_percentile, threshold_adaptive)

        # If max threshold is too low, use absolute minimum (3 G) for realistic event detection
        threshold = max(threshold, 3.0)

        # Get indices of significant events
        significant_indices = amplitude_values_full[amplitude_values_full >= threshold].index

        # Safety check: ensure we have enough event points
        if len(significant_indices) < 10:
            # Fallback: use top 15% of amplitude values
            threshold = amplitude_values_full.quantile(0.85)
            significant_indices = amplitude_values_full[amplitude_values_full >= threshold].index

        # Extract only significant event amplitudes (matching CSV event-level format)
        amplitude_values = amplitude_values_full.loc[significant_indices]

        if len(amplitude_values) == 0:
            # Final fallback: use all values if filtering removes everything
            amplitude_values = amplitude_values_full

        # Calculate 8 features from event-level data
        features = {
            'mean_amplitude': float(amplitude_values.mean()),
            'max_amplitude': float(amplitude_values.max()),
            'min_amplitude': float(amplitude_values.min()),
            'std_amplitude': float(amplitude_values.std()),
            'amplitude_range': float(amplitude_values.max() - amplitude_values.min()),
            'median_amplitude': float(amplitude_values.median()),
            'crank_angle_at_max': float(df.loc[amplitude_values.idxmax(), 'Crank Angle']),
            'sample_count': int(len(amplitude_values))
        }

        return features

    except Exception as e:
        print(f"Failed to extract features: {e}")
        return None


def get_curve_info(xml_content: str) -> Dict[str, any]:
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
