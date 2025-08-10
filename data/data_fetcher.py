"""
Data Fetcher for Financial Market Data
Uses free APIs to gather real-time financial information
"""

import yfinance as yf
import pandas as pd
import requests
from typing import Dict, Optional, List
import os
from datetime import datetime, timedelta
import numpy as np


class DataFetcher:
    """
    Fetches financial data from various free APIs
    """
    
    def __init__(self):
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.fred_key = os.getenv('FRED_API_KEY')
        
    def get_company_info(self, ticker: str) -> Optional[Dict]:
        """
        Get company information using Yahoo Finance
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with company information
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Clean and standardize the data
            company_data = {
                'symbol': ticker,
                'shortName': info.get('shortName', ticker),
                'longName': info.get('longName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'marketCap': info.get('marketCap', 0),
                'enterpriseValue': info.get('enterpriseValue', 0),
                'totalRevenue': info.get('totalRevenue', 0),
                'ebitda': info.get('ebitda', 0),
                'grossProfits': info.get('grossProfits', 0),
                'netIncomeToCommon': info.get('netIncomeToCommon', 0),
                'totalDebt': info.get('totalDebt', 0),
                'totalCash': info.get('totalCash', 0),
                'trailingPE': info.get('trailingPE', 0),
                'forwardPE': info.get('forwardPE', 0),
                'priceToBook': info.get('priceToBook', 0),
                'returnOnEquity': info.get('returnOnEquity', 0),
                'returnOnAssets': info.get('returnOnAssets', 0),
                'debtToEquity': info.get('debtToEquity', 0),
                'currentRatio': info.get('currentRatio', 0),
                'quickRatio': info.get('quickRatio', 0),
                'beta': info.get('beta', 0),
                'dividendYield': info.get('dividendYield', 0),
                'payoutRatio': info.get('payoutRatio', 0),
                'lastFiscalYearEnd': info.get('lastFiscalYearEnd', 0),
                'nextFiscalYearEnd': info.get('nextFiscalYearEnd', 0),
                'mostRecentQuarter': info.get('mostRecentQuarter', 0)
            }
            
            return company_data
            
        except Exception as e:
            print(f"Error fetching company info for {ticker}: {e}")
            return None
    
    def get_stock_data(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """
        Get historical stock price data
        
        Args:
            ticker: Stock ticker symbol
            period: Time period ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            DataFrame with stock price data
        """
        try:
            stock = yf.Ticker(ticker)
            hist = stock.history(period=period)
            
            # If 1 year data is not available, try shorter periods
            if hist.empty and period == "1y":
                for fallback_period in ["6mo", "3mo", "1mo"]:
                    hist = stock.history(period=fallback_period)
                    if not hist.empty:
                        break
            
            return hist
        except Exception as e:
            print(f"Error fetching stock data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """
        Get financial statements for a company
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with income statement, balance sheet, and cash flow
        """
        try:
            stock = yf.Ticker(ticker)
            
            financials = {
                'income_statement': stock.financials,
                'balance_sheet': stock.balance_sheet,
                'cash_flow': stock.cashflow
            }
            
            return financials
            
        except Exception as e:
            print(f"Error fetching financial statements for {ticker}: {e}")
            return {}
    
    def get_current_price(self, ticker: str) -> Optional[float]:
        """
        Get current stock price
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Current stock price
        """
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(period="1d")
            if not data.empty:
                return data['Close'].iloc[-1]
            return None
        except Exception as e:
            print(f"Error fetching current price for {ticker}: {e}")
            return None
    
    def get_market_data(self) -> Dict:
        """
        Get general market data and economic indicators
        
        Returns:
            Dictionary with market indicators
        """
        try:
            # Get major indices
            indices = {
                'SPY': '^GSPC',  # S&P 500
                'QQQ': '^IXIC',  # NASDAQ
                'DJI': '^DJI',   # Dow Jones
                'VIX': '^VIX'    # Volatility Index
            }
            
            market_data = {}
            
            for name, symbol in indices.items():
                try:
                    ticker_data = yf.Ticker(symbol)
                    hist = ticker_data.history(period="5d")
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        prev_price = hist['Close'].iloc[-2] if len(hist) > 1 else current_price
                        change_pct = ((current_price - prev_price) / prev_price) * 100
                        
                        market_data[name] = {
                            'price': current_price,
                            'change_pct': change_pct
                        }
                except:
                    continue
            
            # Get Treasury rates (10-year as proxy for risk-free rate)
            try:
                treasury = yf.Ticker("^TNX")
                treasury_data = treasury.history(period="5d")
                if not treasury_data.empty:
                    market_data['10Y_Treasury'] = {
                        'rate': treasury_data['Close'].iloc[-1] / 100  # Convert to decimal
                    }
            except:
                market_data['10Y_Treasury'] = {'rate': 0.045}  # Default 4.5%
            
            return market_data
            
        except Exception as e:
            print(f"Error fetching market data: {e}")
            return {}
    
    def get_sector_performance(self, sectors: List[str]) -> pd.DataFrame:
        """
        Get sector performance data
        
        Args:
            sectors: List of sector ETF symbols
            
        Returns:
            DataFrame with sector performance
        """
        try:
            # Common sector ETFs
            sector_etfs = {
                'Technology': 'XLK',
                'Healthcare': 'XLV',
                'Financials': 'XLF',
                'Consumer Discretionary': 'XLY',
                'Consumer Staples': 'XLP',
                'Energy': 'XLE',
                'Industrials': 'XLI',
                'Materials': 'XLB',
                'Real Estate': 'XLRE',
                'Utilities': 'XLU',
                'Communication': 'XLC'
            }
            
            sector_data = []
            
            for sector_name, etf_symbol in sector_etfs.items():
                try:
                    etf = yf.Ticker(etf_symbol)
                    hist = etf.history(period="1y")
                    
                    if not hist.empty:
                        current_price = hist['Close'].iloc[-1]
                        year_ago_price = hist['Close'].iloc[0]
                        ytd_return = ((current_price - year_ago_price) / year_ago_price) * 100
                        
                        # Get recent performance
                        if len(hist) >= 21:  # 1 month
                            month_ago_price = hist['Close'].iloc[-21]
                            monthly_return = ((current_price - month_ago_price) / month_ago_price) * 100
                        else:
                            monthly_return = 0
                        
                        sector_data.append({
                            'Sector': sector_name,
                            'ETF_Symbol': etf_symbol,
                            'Current_Price': round(current_price, 2),
                            'YTD_Return_%': round(ytd_return, 2),
                            'Monthly_Return_%': round(monthly_return, 2)
                        })
                        
                except Exception as e:
                    print(f"Error fetching data for {sector_name}: {e}")
                    continue
            
            return pd.DataFrame(sector_data)
            
        except Exception as e:
            print(f"Error fetching sector performance: {e}")
            return pd.DataFrame()
    
    def get_peer_companies(self, ticker: str, sector: str, max_peers: int = 10) -> List[str]:
        """
        Get peer companies based on sector (simplified approach)
        
        Args:
            ticker: Target company ticker
            sector: Company sector
            max_peers: Maximum number of peers to return
            
        Returns:
            List of peer company tickers
        """
        
        # Predefined peer groups by sector (in practice, you'd use a more sophisticated approach)
        peer_groups = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX', 'NVDA', 'ADBE', 'CRM', 'ORCL', 'IBM'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO', 'DHR', 'MDT', 'BMY', 'AMGN', 'GILD'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS', 'MS', 'USB', 'PNC', 'TFC', 'COF'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE', 'SBUX', 'TJX', 'LOW', 'TGT', 'CMG'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB', 'PSX', 'VLO', 'MPC', 'OXY', 'HAL'],
            'Industrials': ['BA', 'CAT', 'GE', 'MMM', 'HON', 'UPS', 'LMT', 'RTX', 'DE', 'EMR'],
            'Consumer Defensive': ['PG', 'KO', 'PEP', 'WMT', 'COST', 'CL', 'KMB', 'GIS', 'K', 'CPB']
        }
        
        # Get peers for the sector, excluding the target company
        peers = peer_groups.get(sector, [])
        filtered_peers = [peer for peer in peers if peer != ticker.upper()]
        
        return filtered_peers[:max_peers]
    
    def get_economic_indicators(self) -> Dict:
        """
        Get key economic indicators (using available free sources)
        
        Returns:
            Dictionary with economic indicators
        """
        try:
            indicators = {}
            
            # GDP Growth (simplified - using market proxy)
            try:
                spy = yf.Ticker("SPY")
                spy_hist = spy.history(period="1y")
                if not spy_hist.empty:
                    current_spy = spy_hist['Close'].iloc[-1]
                    year_ago_spy = spy_hist['Close'].iloc[0]
                    market_growth = ((current_spy - year_ago_spy) / year_ago_spy) * 100
                    indicators['Market_Growth_%'] = round(market_growth, 2)
            except:
                indicators['Market_Growth_%'] = 8.5  # Default assumption
            
            # Risk-free rate (10-year Treasury)
            try:
                treasury = yf.Ticker("^TNX")
                treasury_hist = treasury.history(period="5d")
                if not treasury_hist.empty:
                    risk_free_rate = treasury_hist['Close'].iloc[-1] / 100
                    indicators['Risk_Free_Rate'] = round(risk_free_rate, 4)
            except:
                indicators['Risk_Free_Rate'] = 0.045  # Default 4.5%
            
            # Market volatility
            try:
                vix = yf.Ticker("^VIX")
                vix_hist = vix.history(period="5d")
                if not vix_hist.empty:
                    volatility = vix_hist['Close'].iloc[-1]
                    indicators['Market_Volatility'] = round(volatility, 2)
            except:
                indicators['Market_Volatility'] = 20.0  # Default assumption
            
            # Currency (USD strength via DXY proxy)
            try:
                dxy = yf.Ticker("DX-Y.NYB")
                dxy_hist = dxy.history(period="1mo")
                if not dxy_hist.empty:
                    current_dxy = dxy_hist['Close'].iloc[-1]
                    month_ago_dxy = dxy_hist['Close'].iloc[0]
                    usd_change = ((current_dxy - month_ago_dxy) / month_ago_dxy) * 100
                    indicators['USD_Strength_%'] = round(usd_change, 2)
            except:
                indicators['USD_Strength_%'] = 0.5  # Default assumption
            
            return indicators
            
        except Exception as e:
            print(f"Error fetching economic indicators: {e}")
            return {
                'Market_Growth_%': 8.5,
                'Risk_Free_Rate': 0.045,
                'Market_Volatility': 20.0,
                'USD_Strength_%': 0.5
            }
