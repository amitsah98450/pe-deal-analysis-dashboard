#!/usr/bin/env python3

import sys
import subprocess

print("Testing Python environment...")
print(f"Python version: {sys.version}")
print(f"Python executable: {sys.executable}")

# Test basic imports
try:
    import streamlit
    print(f"✅ Streamlit version: {streamlit.__version__}")
except ImportError as e:
    print(f"❌ Streamlit not found: {e}")

try:
    import pandas
    print(f"✅ Pandas version: {pandas.__version__}")
except ImportError as e:
    print(f"❌ Pandas not found: {e}")

try:
    import plotly
    print(f"✅ Plotly version: {plotly.__version__}")
except ImportError as e:
    print(f"❌ Plotly not found: {e}")

try:
    import yfinance
    print(f"✅ yfinance imported successfully")
except ImportError as e:
    print(f"❌ yfinance not found: {e}")

try:
    import numpy
    print(f"✅ NumPy version: {numpy.__version__}")
except ImportError as e:
    print(f"❌ NumPy not found: {e}")

print("\nTesting custom modules...")
try:
    from models.dcf_model import DCFModel
    print("✅ DCF Model imported")
except ImportError as e:
    print(f"❌ DCF Model import failed: {e}")

try:
    from data.data_fetcher import DataFetcher
    print("✅ Data Fetcher imported")
except ImportError as e:
    print(f"❌ Data Fetcher import failed: {e}")

print("\nEnvironment test complete!")
