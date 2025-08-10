"""
Valuation Models for Private Equity Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from data.data_fetcher import DataFetcher


class ValuationModel:
    """
    Comprehensive valuation models for private equity analysis
    """
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
    
    def comparable_analysis(self, target_ticker: str, peer_tickers: List[str]) -> pd.DataFrame:
        """
        Perform comparable company analysis
        
        Args:
            target_ticker: Target company ticker
            peer_tickers: List of peer company tickers
            
        Returns:
            DataFrame with comparable metrics
        """
        
        companies = [target_ticker] + peer_tickers
        comp_data = []
        
        for ticker in companies:
            try:
                company_info = self.data_fetcher.get_company_info(ticker)
                if company_info:
                    # Extract key metrics
                    market_cap = company_info.get('marketCap', 0) / 1e9  # Convert to billions
                    enterprise_value = company_info.get('enterpriseValue', 0) / 1e9
                    revenue = company_info.get('totalRevenue', 0) / 1e9
                    ebitda = company_info.get('ebitda', 0) / 1e9
                    net_income = company_info.get('netIncomeToCommon', 0) / 1e9
                    
                    # Calculate multiples
                    pe_ratio = company_info.get('trailingPE', 0)
                    ev_revenue = enterprise_value / revenue if revenue > 0 else 0
                    ev_ebitda = enterprise_value / ebitda if ebitda > 0 else 0
                    
                    # Additional metrics
                    roe = company_info.get('returnOnEquity', 0) * 100 if company_info.get('returnOnEquity') else 0
                    debt_to_equity = company_info.get('debtToEquity', 0) / 100 if company_info.get('debtToEquity') else 0
                    
                    comp_data.append({
                        'Company': ticker,
                        'Market Cap ($B)': round(market_cap, 2),
                        'Enterprise Value ($B)': round(enterprise_value, 2),
                        'Revenue ($B)': round(revenue, 2),
                        'EBITDA ($B)': round(ebitda, 2),
                        'P/E Ratio': round(pe_ratio, 1),
                        'EV/Revenue': round(ev_revenue, 1),
                        'EV/EBITDA': round(ev_ebitda, 1),
                        'ROE (%)': round(roe, 1),
                        'Debt/Equity': round(debt_to_equity, 2)
                    })
                    
            except Exception as e:
                print(f"Error fetching data for {ticker}: {e}")
                continue
        
        return pd.DataFrame(comp_data)
    
    def precedent_transactions(self, sector: str, deal_size_range: Tuple[float, float]) -> pd.DataFrame:
        """
        Create precedent transaction analysis (simulated data for demo)
        
        Args:
            sector: Industry sector
            deal_size_range: Min and max deal size in billions
            
        Returns:
            DataFrame with precedent transaction multiples
        """
        
        # Simulated precedent transaction data
        np.random.seed(42)  # For reproducible results
        
        num_transactions = 20
        
        # Generate random transaction data
        deal_sizes = np.random.uniform(deal_size_range[0], deal_size_range[1], num_transactions)
        ev_revenue_multiples = np.random.normal(2.5, 0.8, num_transactions)
        ev_ebitda_multiples = np.random.normal(12.0, 3.0, num_transactions)
        
        # Adjust multiples based on sector
        sector_adjustments = {
            'Technology': {'ev_revenue': 1.5, 'ev_ebitda': 1.2},
            'Healthcare': {'ev_revenue': 1.2, 'ev_ebitda': 1.1},
            'Financial Services': {'ev_revenue': 0.8, 'ev_ebitda': 0.9},
            'Consumer': {'ev_revenue': 1.0, 'ev_ebitda': 1.0},
            'Energy': {'ev_revenue': 0.7, 'ev_ebitda': 0.8}
        }
        
        adjustment = sector_adjustments.get(sector, {'ev_revenue': 1.0, 'ev_ebitda': 1.0})
        ev_revenue_multiples *= adjustment['ev_revenue']
        ev_ebitda_multiples *= adjustment['ev_ebitda']
        
        # Create transaction DataFrame
        transactions = []
        for i in range(num_transactions):
            year = np.random.choice([2021, 2022, 2023, 2024])
            
            transactions.append({
                'Deal #': f"Deal {i+1}",
                'Year': year,
                'Enterprise Value ($B)': round(deal_sizes[i], 2),
                'EV/Revenue': round(max(0.1, ev_revenue_multiples[i]), 1),
                'EV/EBITDA': round(max(5.0, ev_ebitda_multiples[i]), 1),
                'Sector': sector
            })
        
        precedent_df = pd.DataFrame(transactions)
        return precedent_df.sort_values('Year', ascending=False)
    
    def sum_of_parts_valuation(self, business_segments: List[Dict]) -> Dict:
        """
        Perform sum-of-parts valuation for multi-segment companies
        
        Args:
            business_segments: List of business segment dictionaries with metrics
            
        Returns:
            Dictionary with sum-of-parts valuation results
        """
        
        total_value = 0
        segment_values = []
        
        for segment in business_segments:
            name = segment['name']
            revenue = segment['revenue']
            ebitda = segment['ebitda']
            multiple = segment.get('ev_ebitda_multiple', 10.0)
            
            segment_value = ebitda * multiple
            total_value += segment_value
            
            segment_values.append({
                'Segment': name,
                'Revenue ($M)': revenue,
                'EBITDA ($M)': ebitda,
                'EV/EBITDA Multiple': multiple,
                'Segment Value ($M)': round(segment_value, 1),
                'Value Weight (%)': 0  # Will be calculated after total
            })
        
        # Calculate weights
        for segment in segment_values:
            segment['Value Weight (%)'] = round(
                (segment['Segment Value ($M)'] / total_value) * 100, 1
            )
        
        return {
            'total_enterprise_value': total_value,
            'segment_valuations': segment_values,
            'value_per_segment': {seg['Segment']: seg['Segment Value ($M)'] for seg in segment_values}
        }
    
    def net_asset_value(self, assets: List[Dict], liabilities: List[Dict]) -> Dict:
        """
        Calculate Net Asset Value for asset-heavy businesses
        
        Args:
            assets: List of asset dictionaries with values
            liabilities: List of liability dictionaries with values
            
        Returns:
            Dictionary with NAV calculation
        """
        
        total_assets = sum(asset['value'] for asset in assets)
        total_liabilities = sum(liability['value'] for liability in liabilities)
        
        nav = total_assets - total_liabilities
        
        # Create detailed breakdown
        asset_breakdown = pd.DataFrame(assets)
        liability_breakdown = pd.DataFrame(liabilities)
        
        return {
            'net_asset_value': nav,
            'total_assets': total_assets,
            'total_liabilities': total_liabilities,
            'asset_breakdown': asset_breakdown,
            'liability_breakdown': liability_breakdown
        }
    
    def dividend_discount_model(self, current_dividend: float, growth_rate: float,
                              required_return: float, growth_phases: Optional[List[Dict]] = None) -> Dict:
        """
        Calculate valuation using Dividend Discount Model
        
        Args:
            current_dividend: Current annual dividend per share
            growth_rate: Long-term dividend growth rate
            required_return: Required rate of return
            growth_phases: Optional list of different growth phases
            
        Returns:
            Dictionary with DDM valuation
        """
        
        if growth_phases:
            # Multi-stage DDM
            total_pv = 0
            dividends = []
            year = 0
            
            for phase in growth_phases:
                phase_years = phase['years']
                phase_growth = phase['growth_rate']
                
                for i in range(phase_years):
                    year += 1
                    if year == 1:
                        dividend = current_dividend * (1 + phase_growth)
                    else:
                        dividend = dividends[-1] * (1 + phase_growth)
                    
                    pv_dividend = dividend / (1 + required_return) ** year
                    dividends.append(dividend)
                    total_pv += pv_dividend
            
            # Terminal value
            terminal_dividend = dividends[-1] * (1 + growth_rate)
            terminal_value = terminal_dividend / (required_return - growth_rate)
            pv_terminal = terminal_value / (1 + required_return) ** year
            
            total_value = total_pv + pv_terminal
            
            return {
                'value_per_share': total_value,
                'pv_dividends': total_pv,
                'terminal_value': terminal_value,
                'pv_terminal_value': pv_terminal,
                'dividend_projections': dividends
            }
        
        else:
            # Gordon Growth Model (single stage)
            next_dividend = current_dividend * (1 + growth_rate)
            value_per_share = next_dividend / (required_return - growth_rate)
            
            return {
                'value_per_share': value_per_share,
                'next_dividend': next_dividend,
                'growth_rate': growth_rate,
                'required_return': required_return
            }
    
    def option_valuation_real_options(self, underlying_value: float, strike_price: float,
                                    volatility: float, risk_free_rate: float,
                                    time_to_expiry: float) -> Dict:
        """
        Value real options using Black-Scholes framework
        
        Args:
            underlying_value: Present value of underlying asset/project
            strike_price: Investment required to exercise option
            volatility: Volatility of underlying asset value
            risk_free_rate: Risk-free interest rate
            time_to_expiry: Time to option expiry in years
            
        Returns:
            Dictionary with option valuation
        """
        
        from scipy.stats import norm
        import math
        
        # Black-Scholes calculation
        d1 = (math.log(underlying_value / strike_price) + 
              (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
        
        d2 = d1 - volatility * math.sqrt(time_to_expiry)
        
        # Call option value (real option value)
        option_value = (underlying_value * norm.cdf(d1) - 
                       strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        
        # Greeks
        delta = norm.cdf(d1)
        gamma = norm.pdf(d1) / (underlying_value * volatility * math.sqrt(time_to_expiry))
        theta = (-(underlying_value * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry)) -
                risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        vega = underlying_value * norm.pdf(d1) * math.sqrt(time_to_expiry)
        
        return {
            'option_value': option_value,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'd1': d1,
            'd2': d2,
            'intrinsic_value': max(0, underlying_value - strike_price),
            'time_value': option_value - max(0, underlying_value - strike_price)
        }
