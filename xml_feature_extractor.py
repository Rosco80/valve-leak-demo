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

        # Extract data (rows 6 onwards)
        data = [[cell.find('ss:Data', XML_NS).text for cell in r.findall('ss:Cell', XML_NS)] for r in rows[6:]]

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
            # Look for ULTRASONIC or AE column
            ae_columns = [col for col in df.columns
                         if col != 'Crank Angle' and
                         ('ULTRASONIC' in col.upper() or 'AE' in col.upper() or 'KHZ' in col.upper())]

            if not ae_columns:
                # No AE column found, use first non-crank-angle column
                ae_columns = [col for col in df.columns if col != 'Crank Angle']

            if not ae_columns:
                return None

            curve_name = ae_columns[0]

        # Extract amplitude values
        amplitude_values = df[curve_name].dropna()

        if len(amplitude_values) == 0:
            return None

        # Calculate 8 features
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
