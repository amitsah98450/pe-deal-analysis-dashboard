"""
Market Data Processing and Analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import yfinance as yf


class MarketDataProcessor:
    """
    Process and analyze market data for private equity analysis
    """
    
    def __init__(self):
        pass
    
    def calculate_beta(self, stock_returns: pd.Series, market_returns: pd.Series) -> float:
        """
        Calculate beta coefficient for a stock
        
        Args:
            stock_returns: Stock return series
            market_returns: Market return series
            
        Returns:
            Beta coefficient
        """
        try:
            # Align the series
            aligned_data = pd.DataFrame({
                'stock': stock_returns,
                'market': market_returns
            }).dropna()
            
            if len(aligned_data) < 10:  # Need sufficient data
                return 1.0  # Default beta
            
            covariance = aligned_data['stock'].cov(aligned_data['market'])
            market_variance = aligned_data['market'].var()
            
            if market_variance == 0:
                return 1.0
            
            beta = covariance / market_variance
            return beta
            
        except Exception:
            return 1.0  # Default beta if calculation fails
    
    def calculate_volatility(self, returns: pd.Series, annualize: bool = True) -> float:
        """
        Calculate volatility of returns
        
        Args:
            returns: Return series
            annualize: Whether to annualize the volatility
            
        Returns:
            Volatility (standard deviation)
        """
        try:
            vol = returns.std()
            if annualize:
                # Annualize assuming 252 trading days
                vol = vol * np.sqrt(252)
            return vol
        except Exception:
            return 0.20  # Default 20% volatility
    
    def calculate_sharpe_ratio(self, returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio
        
        Args:
            returns: Return series
            risk_free_rate: Risk-free rate (annual)
            
        Returns:
            Sharpe ratio
        """
        try:
            excess_returns = returns - risk_free_rate / 252  # Daily risk-free rate
            if len(excess_returns) == 0 or excess_returns.std() == 0:
                return 0.0
            
            sharpe = excess_returns.mean() / excess_returns.std() * np.sqrt(252)
            return sharpe
        except Exception:
            return 0.0
    
    def calculate_maximum_drawdown(self, price_series: pd.Series) -> Dict[str, float]:
        """
        Calculate maximum drawdown and related metrics
        
        Args:
            price_series: Price series
            
        Returns:
            Dictionary with drawdown metrics
        """
        try:
            # Calculate cumulative returns
            cumulative = (1 + price_series.pct_change()).cumprod()
            
            # Calculate running maximum
            running_max = cumulative.expanding().max()
            
            # Calculate drawdown
            drawdown = (cumulative - running_max) / running_max
            
            max_drawdown = drawdown.min()
            max_drawdown_duration = 0
            current_duration = 0
            
            # Calculate maximum drawdown duration
            for dd in drawdown:
                if dd < 0:
                    current_duration += 1
                    max_drawdown_duration = max(max_drawdown_duration, current_duration)
                else:
                    current_duration = 0
            
            return {
                'max_drawdown': max_drawdown,
                'max_drawdown_duration': max_drawdown_duration,
                'current_drawdown': drawdown.iloc[-1] if len(drawdown) > 0 else 0
            }
            
        except Exception:
            return {
                'max_drawdown': -0.20,  # Default -20%
                'max_drawdown_duration': 30,
                'current_drawdown': 0
            }
    
    def analyze_correlation_matrix(self, tickers: List[str], period: str = "1y") -> pd.DataFrame:
        """
        Create correlation matrix for multiple stocks
        
        Args:
            tickers: List of stock tickers
            period: Time period for analysis
            
        Returns:
            Correlation matrix DataFrame
        """
        try:
            returns_data = {}
            
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    returns_data[ticker] = returns
            
            if returns_data:
                returns_df = pd.DataFrame(returns_data)
                correlation_matrix = returns_df.corr()
                return correlation_matrix
            else:
                return pd.DataFrame()
                
        except Exception:
            return pd.DataFrame()
    
    def sector_rotation_analysis(self, sector_etfs: Dict[str, str], period: str = "1y") -> pd.DataFrame:
        """
        Analyze sector rotation patterns
        
        Args:
            sector_etfs: Dictionary mapping sector names to ETF tickers
            period: Analysis period
            
        Returns:
            DataFrame with sector performance analysis
        """
        try:
            sector_performance = []
            
            for sector, etf in sector_etfs.items():
                try:
                    ticker = yf.Ticker(etf)
                    hist = ticker.history(period=period)
                    
                    if not hist.empty:
                        returns = hist['Close'].pct_change().dropna()
                        
                        # Calculate metrics
                        total_return = (hist['Close'].iloc[-1] / hist['Close'].iloc[0] - 1) * 100
                        volatility = self.calculate_volatility(returns) * 100
                        sharpe = self.calculate_sharpe_ratio(returns)
                        max_dd = self.calculate_maximum_drawdown(hist['Close'])['max_drawdown'] * 100
                        
                        sector_performance.append({
                            'Sector': sector,
                            'ETF': etf,
                            'Total_Return_%': round(total_return, 2),
                            'Volatility_%': round(volatility, 2),
                            'Sharpe_Ratio': round(sharpe, 2),
                            'Max_Drawdown_%': round(max_dd, 2)
                        })
                        
                except Exception:
                    continue
            
            return pd.DataFrame(sector_performance)
            
        except Exception:
            return pd.DataFrame()
    
    def calculate_var_cvar(self, returns: pd.Series, confidence_level: float = 0.05) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR)
        
        Args:
            returns: Return series
            confidence_level: Confidence level (0.05 for 95% VaR)
            
        Returns:
            Dictionary with VaR and CVaR
        """
        try:
            if len(returns) == 0:
                return {'var': 0, 'cvar': 0}
            
            # Sort returns
            sorted_returns = returns.sort_values()
            
            # Calculate VaR
            var_index = int(confidence_level * len(sorted_returns))
            var = sorted_returns.iloc[var_index] if var_index < len(sorted_returns) else sorted_returns.iloc[0]
            
            # Calculate CVaR (expected shortfall)
            cvar_returns = sorted_returns.iloc[:var_index] if var_index > 0 else sorted_returns.iloc[:1]
            cvar = cvar_returns.mean()
            
            return {
                'var': var,
                'cvar': cvar,
                'var_percentage': var * 100,
                'cvar_percentage': cvar * 100
            }
            
        except Exception:
            return {'var': -0.05, 'cvar': -0.08, 'var_percentage': -5.0, 'cvar_percentage': -8.0}
    
    def momentum_analysis(self, price_series: pd.Series, periods: List[int] = [20, 50, 200]) -> Dict:
        """
        Analyze price momentum using moving averages
        
        Args:
            price_series: Price series
            periods: List of periods for moving averages
            
        Returns:
            Dictionary with momentum indicators
        """
        try:
            momentum_data = {}
            current_price = price_series.iloc[-1]
            
            for period in periods:
                if len(price_series) >= period:
                    ma = price_series.rolling(window=period).mean().iloc[-1]
                    momentum_data[f'MA_{period}'] = ma
                    momentum_data[f'Price_vs_MA_{period}_%'] = ((current_price - ma) / ma) * 100
                    
                    # Trend direction
                    if len(price_series) >= period + 5:
                        ma_prev = price_series.rolling(window=period).mean().iloc[-6]
                        momentum_data[f'MA_{period}_Trend'] = 'Up' if ma > ma_prev else 'Down'
            
            # RSI calculation (simplified)
            if len(price_series) >= 14:
                delta = price_series.diff()
                gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs))
                momentum_data['RSI'] = rsi.iloc[-1]
            
            return momentum_data
            
        except Exception:
            return {}
    
    def market_regime_detection(self, market_returns: pd.Series, window: int = 63) -> pd.Series:
        """
        Detect market regimes based on volatility
        
        Args:
            market_returns: Market return series
            window: Rolling window for volatility calculation
            
        Returns:
            Series with regime labels
        """
        try:
            # Calculate rolling volatility
            rolling_vol = market_returns.rolling(window=window).std() * np.sqrt(252)
            
            # Define regime thresholds (can be calibrated)
            low_vol_threshold = rolling_vol.quantile(0.33)
            high_vol_threshold = rolling_vol.quantile(0.67)
            
            # Assign regimes
            regimes = pd.Series(index=rolling_vol.index, dtype=str)
            regimes[rolling_vol <= low_vol_threshold] = 'Low Volatility'
            regimes[(rolling_vol > low_vol_threshold) & (rolling_vol <= high_vol_threshold)] = 'Medium Volatility'
            regimes[rolling_vol > high_vol_threshold] = 'High Volatility'
            
            return regimes
            
        except Exception:
            return pd.Series()
    
    def create_factor_returns(self, tickers: List[str], factors: Dict[str, str], period: str = "2y") -> pd.DataFrame:
        """
        Calculate factor exposures and returns
        
        Args:
            tickers: List of stock tickers
            factors: Dictionary of factor names and their proxy tickers
            period: Analysis period
            
        Returns:
            DataFrame with factor loadings
        """
        try:
            all_data = {}
            
            # Fetch stock data
            for ticker in tickers:
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    all_data[ticker] = returns
            
            # Fetch factor data
            for factor_name, factor_ticker in factors.items():
                factor = yf.Ticker(factor_ticker)
                hist = factor.history(period=period)
                if not hist.empty:
                    returns = hist['Close'].pct_change().dropna()
                    all_data[factor_name] = returns
            
            if all_data:
                returns_df = pd.DataFrame(all_data).dropna()
                
                # Calculate factor loadings using regression
                factor_loadings = {}
                factor_names = list(factors.keys())
                
                for ticker in tickers:
                    if ticker in returns_df.columns:
                        y = returns_df[ticker]
                        X = returns_df[factor_names]
                        
                        # Simple linear regression
                        try:
                            from sklearn.linear_model import LinearRegression
                            model = LinearRegression()
                            model.fit(X, y)
                            
                            loadings = dict(zip(factor_names, model.coef_))
                            loadings['R_squared'] = model.score(X, y)
                            factor_loadings[ticker] = loadings
                            
                        except ImportError:
                            # Fallback to correlation if sklearn not available
                            correlations = {}
                            for factor in factor_names:
                                correlations[factor] = returns_df[ticker].corr(returns_df[factor])
                            factor_loadings[ticker] = correlations
                
                return pd.DataFrame(factor_loadings).T
            
            return pd.DataFrame()
            
        except Exception:
            return pd.DataFrame()
