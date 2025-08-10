# Private Equity Deal Analysis Dashboard

A comprehensive financial analysis dashboard for private equity deal evaluation, built with Python and Streamlit.

## Features

- **DCF Modeling**: Discounted Cash Flow analysis with sensitivity testing
- **LBO Modeling**: Leveraged Buyout models with multiple scenarios
- **Company Valuation**: Comparative analysis using multiple methodologies
- **Industry Benchmarking**: Peer comparison and industry metrics
- **Risk Assessment**: Monte Carlo simulations and stress testing
- **Real-time Data**: Integration with financial APIs for live market data

## Installation

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables (optional):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

## Project Structure

```
├── app.py                 # Main Streamlit application
├── models/               # Financial modeling modules
│   ├── dcf_model.py      # DCF calculations
│   ├── lbo_model.py      # LBO modeling
│   └── valuation.py      # Valuation methods
├── data/                 # Data processing and API integration
│   ├── data_fetcher.py   # API data collection
│   └── market_data.py    # Market data processing
├── utils/                # Utility functions
│   ├── calculations.py   # Financial calculations
│   └── visualizations.py # Chart generation
└── config/               # Configuration files
    └── settings.py       # App settings
```

## API Keys (Optional)

For enhanced functionality, you can add API keys:
- Alpha Vantage API (free tier available)
- FRED (Federal Reserve Economic Data)

## License

This project is for educational and demonstration purposes.
