"""
Health check for the PE Deal Analysis Dashboard
"""

import streamlit as st

def health_check():
    """Simple health check function"""
    try:
        # Test basic imports
        import numpy as np
        import pandas as pd
        import plotly.express as px
        
        # Test custom modules
        from models.dcf_model import DCFModel
        from data.data_fetcher import DataFetcher
        
        return True
    except Exception as e:
        st.error(f"Health check failed: {e}")
        return False

if __name__ == "__main__":
    if health_check():
        st.success("✅ Application health check passed!")
    else:
        st.error("❌ Application health check failed!")
