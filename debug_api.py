#!/usr/bin/env python3
"""Debug script to test the FeatureAPI integration."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'data', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

from data.src.api.feature_api import FeatureAPI

def test_api_integration():
    """Test the API integration to see what's failing."""
    api = FeatureAPI()

    request_data = {
        'data': {
            'price': [100.0, 101.0, 102.0],
            'volume': [1000, 1100, 1200],
            'dates': ['2023-01-01', '2023-01-02', '2023-01-03']
        },
        'features': ['returns'],
        'config': {}
    }

    response = api.generate_features_endpoint(request_data)
    print("Response:", response)
    print("Status:", response.get('status'))
    print("Message:", response.get('message'))

    return response

if __name__ == "__main__":
    test_api_integration()