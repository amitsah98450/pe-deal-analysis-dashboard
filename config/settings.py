"""
Application Configuration Settings
"""

import os
from typing import Dict, Any


# Application Configuration
APP_CONFIG = {
    'app_name': 'PE Deal Analysis Dashboard',
    'version': '1.0.0',
    'debug': os.getenv('DEBUG', 'False').lower() == 'true',
    'port': int(os.getenv('PORT', 8501)),
    'host': os.getenv('HOST', '0.0.0.0')
}

# Financial Model Defaults
FINANCIAL_DEFAULTS = {
    'projection_years': 5,
    'tax_rate': 0.25,
    'terminal_growth_rate': 0.025,
    'wacc': 0.08,
    'risk_free_rate': 0.045,
    'market_risk_premium': 0.06,
    'default_beta': 1.0
}

# LBO Model Defaults
LBO_DEFAULTS = {
    'hold_period': 5,
    'management_rollover': 0.05,
    'transaction_fees': 0.025,
    'cash_sweep_ratio': 0.75,
    'minimum_cash_balance': 50,  # millions
    'interest_rate': 0.06
}

# Valuation Multiples by Sector
SECTOR_MULTIPLES = {
    'Technology': {
        'ev_revenue_median': 4.5,
        'ev_ebitda_median': 18.0,
        'pe_median': 25.0
    },
    'Healthcare': {
        'ev_revenue_median': 3.2,
        'ev_ebitda_median': 15.5,
        'pe_median': 18.5
    },
    'Financial Services': {
        'ev_revenue_median': 2.1,
        'ev_ebitda_median': 9.5,
        'pe_median': 12.0
    },
    'Consumer Cyclical': {
        'ev_revenue_median': 1.8,
        'ev_ebitda_median': 12.5,
        'pe_median': 16.0
    },
    'Energy': {
        'ev_revenue_median': 1.2,
        'ev_ebitda_median': 8.5,
        'pe_median': 14.0
    },
    'Industrials': {
        'ev_revenue_median': 2.5,
        'ev_ebitda_median': 13.0,
        'pe_median': 17.5
    },
    'Consumer Defensive': {
        'ev_revenue_median': 2.0,
        'ev_ebitda_median': 11.5,
        'pe_median': 20.0
    }
}

# API Configuration
API_CONFIG = {
    'alpha_vantage': {
        'base_url': 'https://www.alphavantage.co/query',
        'api_key': os.getenv('ALPHA_VANTAGE_API_KEY'),
        'rate_limit': 5  # requests per minute for free tier
    },
    'fred': {
        'base_url': 'https://api.stlouisfed.org/fred',
        'api_key': os.getenv('FRED_API_KEY'),
        'rate_limit': 120  # requests per minute
    }
}

# Chart Configuration
CHART_CONFIG = {
    'default_theme': 'plotly_white',
    'color_palette': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
    'height': {
        'default': 500,
        'dashboard': 400,
        'detailed': 600,
        'full': 800
    },
    'margins': {
        'l': 60,
        'r': 60,
        't': 80,
        'b': 60
    }
}

# Risk Assessment Parameters
RISK_PARAMETERS = {
    'monte_carlo': {
        'default_simulations': 5000,
        'max_simulations': 20000,
        'confidence_levels': [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]
    },
    'stress_scenarios': {
        'mild_stress': {
            'revenue_growth_shock': -0.02,  # -2% from base
            'margin_compression': -0.03,    # -3% from base
            'multiple_compression': -1.0     # -1x from base
        },
        'moderate_stress': {
            'revenue_growth_shock': -0.05,
            'margin_compression': -0.05,
            'multiple_compression': -2.0
        },
        'severe_stress': {
            'revenue_growth_shock': -0.10,
            'margin_compression': -0.08,
            'multiple_compression': -3.0
        }
    }
}

# Data Sources and Peer Groups
PEER_GROUPS = {
    'Technology': {
        'large_cap': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX', 'NVDA'],
        'software': ['CRM', 'ADBE', 'ORCL', 'SAP', 'NOW', 'WDAY'],
        'semiconductors': ['NVDA', 'TSM', 'INTC', 'AMD', 'QCOM', 'AVGO']
    },
    'Healthcare': {
        'pharma': ['JNJ', 'PFE', 'ABBV', 'BMY', 'MRK', 'LLY'],
        'biotech': ['AMGN', 'GILD', 'BIIB', 'VRTX', 'REGN', 'CELG'],
        'devices': ['MDT', 'ABT', 'TMO', 'DHR', 'BSX', 'SYK']
    },
    'Financial Services': {
        'banks': ['JPM', 'BAC', 'WFC', 'C', 'USB', 'PNC'],
        'investment_banks': ['GS', 'MS', 'BLK', 'SCHW', 'SPGI', 'MCO'],
        'insurance': ['BRK.B', 'UNH', 'AIG', 'TRV', 'PGR', 'ALL']
    }
}

# Economic Indicators Mapping
ECONOMIC_INDICATORS = {
    'gdp_growth': '^GSPC',  # S&P 500 as proxy
    'inflation': '^TNX',    # 10-year Treasury as proxy
    'unemployment': 'VIX',  # VIX as economic uncertainty proxy
    'interest_rates': '^TNX'
}

# Streamlit Page Configuration
STREAMLIT_CONFIG = {
    'page_title': 'PE Deal Analysis Dashboard',
    'page_icon': '💼',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
    'menu_items': {
        'Get Help': 'https://github.com/your-repo/pe-dashboard',
        'Report a bug': 'https://github.com/your-repo/pe-dashboard/issues',
        'About': "# Private Equity Deal Analysis Dashboard\nBuilt for comprehensive PE deal evaluation"
    }
}

# Model Validation Rules
VALIDATION_RULES = {
    'dcf': {
        'revenue_growth': {'min': -0.20, 'max': 0.50},
        'ebitda_margin': {'min': 0.01, 'max': 0.60},
        'terminal_growth': {'min': 0.005, 'max': 0.05},
        'wacc': {'min': 0.03, 'max': 0.20}
    },
    'lbo': {
        'purchase_multiple': {'min': 5.0, 'max': 25.0},
        'exit_multiple': {'min': 5.0, 'max': 25.0},
        'debt_multiple': {'min': 1.0, 'max': 10.0},
        'hold_period': {'min': 3, 'max': 10}
    }
}

# Export Settings
EXPORT_CONFIG = {
    'formats': ['xlsx', 'csv', 'pdf'],
    'default_format': 'xlsx',
    'max_file_size': 50 * 1024 * 1024,  # 50MB
    'temp_dir': '/tmp'
}

def get_config(section: str) -> Dict[str, Any]:
    """
    Get configuration for a specific section
    
    Args:
        section: Configuration section name
        
    Returns:
        Configuration dictionary
    """
    config_map = {
        'app': APP_CONFIG,
        'financial': FINANCIAL_DEFAULTS,
        'lbo': LBO_DEFAULTS,
        'multiples': SECTOR_MULTIPLES,
        'api': API_CONFIG,
        'charts': CHART_CONFIG,
        'risk': RISK_PARAMETERS,
        'peers': PEER_GROUPS,
        'indicators': ECONOMIC_INDICATORS,
        'streamlit': STREAMLIT_CONFIG,
        'validation': VALIDATION_RULES,
        'export': EXPORT_CONFIG
    }
    
    return config_map.get(section, {})

def validate_input(value: float, parameter: str, model: str) -> bool:
    """
    Validate user input against defined rules
    
    Args:
        value: Input value to validate
        parameter: Parameter name
        model: Model name (dcf, lbo)
        
    Returns:
        True if valid, False otherwise
    """
    rules = VALIDATION_RULES.get(model, {})
    param_rules = rules.get(parameter, {})
    
    if not param_rules:
        return True
    
    min_val = param_rules.get('min', float('-inf'))
    max_val = param_rules.get('max', float('inf'))
    
    return min_val <= value <= max_val
