#!/bin/bash

# Private Equity Deal Analysis Dashboard Launcher
echo "🏦 Starting Private Equity Deal Analysis Dashboard..."
echo "📊 Loading financial models and data sources..."

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Start Streamlit app
/usr/local/bin/python3 -m streamlit run app.py --server.port 8501 --server.headless true

echo "✅ Dashboard is running on http://localhost:8501"
echo "📈 Ready for PE deal analysis!"
