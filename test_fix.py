#!/usr/bin/env python3
"""Quick test to verify the validation.py fix."""
import sys
sys.path.insert(0, '.')

try:
    import pandas as pd
    from data.validation import validate_daily_sellin, validate_targets, validate_forecast_output
    print("✓ Imports successful")
    
    # Test basic validation function
    print("✓ validation.py fix verified - syntax error resolved")
    
except SyntaxError as e:
    print(f"✗ Syntax error still exists: {e}")
    sys.exit(1)
except Exception as e:
    print(f"✗ Other error: {e}")
    sys.exit(1)