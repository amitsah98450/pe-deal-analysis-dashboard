"""
Financial Calculations and Utilities
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from datetime import datetime, timedelta
import math


class FinancialCalculator:
    """
    Collection of financial calculation functions for private equity analysis
    """
    
    @staticmethod
    def present_value(future_value: float, rate: float, periods: int) -> float:
        """
        Calculate present value
        
        Args:
            future_value: Future cash flow
            rate: Discount rate (per period)
            periods: Number of periods
            
        Returns:
            Present value
        """
        return future_value / ((1 + rate) ** periods)
    
    @staticmethod
    def future_value(present_value: float, rate: float, periods: int) -> float:
        """
        Calculate future value
        
        Args:
            present_value: Present cash flow
            rate: Growth rate (per period)
            periods: Number of periods
            
        Returns:
            Future value
        """
        return present_value * ((1 + rate) ** periods)
    
    @staticmethod
    def npv(cash_flows: List[float], discount_rate: float) -> float:
        """
        Calculate Net Present Value
        
        Args:
            cash_flows: List of cash flows (including initial investment as negative)
            discount_rate: Discount rate
            
        Returns:
            Net Present Value
        """
        npv = 0
        for i, cf in enumerate(cash_flows):
            npv += cf / ((1 + discount_rate) ** i)
        return npv
    
    @staticmethod
    def irr(cash_flows: List[float], guess: float = 0.1, max_iterations: int = 100) -> Optional[float]:
        """
        Calculate Internal Rate of Return using Newton-Raphson method
        
        Args:
            cash_flows: List of cash flows
            guess: Initial guess for IRR
            max_iterations: Maximum iterations
            
        Returns:
            IRR or None if not converged
        """
        try:
            for _ in range(max_iterations):
                npv_val = sum(cf / ((1 + guess) ** i) for i, cf in enumerate(cash_flows))
                npv_derivative = sum(-i * cf / ((1 + guess) ** (i + 1)) for i, cf in enumerate(cash_flows))
                
                if abs(npv_derivative) < 1e-12:
                    break
                
                new_guess = guess - npv_val / npv_derivative
                
                if abs(new_guess - guess) < 1e-8:
                    return new_guess
                
                guess = new_guess
            
            return guess if abs(npv_val) < 1e-6 else None
            
        except Exception:
            return None
    
    @staticmethod
    def calculate_cagr(beginning_value: float, ending_value: float, periods: float) -> float:
        """
        Calculate Compound Annual Growth Rate
        
        Args:
            beginning_value: Starting value
            ending_value: Ending value
            periods: Number of years
            
        Returns:
            CAGR as decimal
        """
        if beginning_value <= 0 or ending_value <= 0 or periods <= 0:
            return 0
        
        return (ending_value / beginning_value) ** (1 / periods) - 1
    
    @staticmethod
    def calculate_roe(net_income: float, shareholders_equity: float) -> float:
        """
        Calculate Return on Equity
        
        Args:
            net_income: Net income
            shareholders_equity: Average shareholders' equity
            
        Returns:
            ROE as decimal
        """
        if shareholders_equity == 0:
            return 0
        return net_income / shareholders_equity
    
    @staticmethod
    def calculate_roic(nopat: float, invested_capital: float) -> float:
        """
        Calculate Return on Invested Capital
        
        Args:
            nopat: Net Operating Profit After Tax
            invested_capital: Average invested capital
            
        Returns:
            ROIC as decimal
        """
        if invested_capital == 0:
            return 0
        return nopat / invested_capital
    
    @staticmethod
    def calculate_ev_ebitda(enterprise_value: float, ebitda: float) -> float:
        """
        Calculate EV/EBITDA multiple
        
        Args:
            enterprise_value: Enterprise value
            ebitda: EBITDA
            
        Returns:
            EV/EBITDA multiple
        """
        if ebitda <= 0:
            return 0
        return enterprise_value / ebitda
    
    @staticmethod
    def calculate_debt_ratios(total_debt: float, total_equity: float, ebitda: float) -> Dict[str, float]:
        """
        Calculate various debt ratios
        
        Args:
            total_debt: Total debt
            total_equity: Total equity
            ebitda: EBITDA
            
        Returns:
            Dictionary with debt ratios
        """
        ratios = {}
        
        # Debt-to-equity ratio
        if total_equity > 0:
            ratios['debt_to_equity'] = total_debt / total_equity
        else:
            ratios['debt_to_equity'] = float('inf')
        
        # Debt-to-capital ratio
        total_capital = total_debt + total_equity
        if total_capital > 0:
            ratios['debt_to_capital'] = total_debt / total_capital
        else:
            ratios['debt_to_capital'] = 0
        
        # Debt-to-EBITDA ratio
        if ebitda > 0:
            ratios['debt_to_ebitda'] = total_debt / ebitda
        else:
            ratios['debt_to_ebitda'] = float('inf')
        
        return ratios
    
    @staticmethod
    def calculate_working_capital_metrics(current_assets: float, current_liabilities: float,
                                        inventory: float, accounts_receivable: float,
                                        accounts_payable: float, revenue: float) -> Dict[str, float]:
        """
        Calculate working capital metrics
        
        Args:
            current_assets: Current assets
            current_liabilities: Current liabilities
            inventory: Inventory
            accounts_receivable: Accounts receivable
            accounts_payable: Accounts payable
            revenue: Annual revenue
            
        Returns:
            Dictionary with working capital metrics
        """
        metrics = {}
        
        # Working capital
        metrics['working_capital'] = current_assets - current_liabilities
        
        # Current ratio
        if current_liabilities > 0:
            metrics['current_ratio'] = current_assets / current_liabilities
        else:
            metrics['current_ratio'] = float('inf')
        
        # Quick ratio
        quick_assets = current_assets - inventory
        if current_liabilities > 0:
            metrics['quick_ratio'] = quick_assets / current_liabilities
        else:
            metrics['quick_ratio'] = float('inf')
        
        # Days calculations (assuming daily revenue)
        if revenue > 0:
            daily_revenue = revenue / 365
            
            # Days Sales Outstanding (DSO)
            metrics['days_sales_outstanding'] = accounts_receivable / daily_revenue
            
            # Days Inventory Outstanding (DIO)
            if inventory > 0:
                daily_cogs = revenue * 0.7 / 365  # Assuming COGS is 70% of revenue
                metrics['days_inventory_outstanding'] = inventory / daily_cogs
            else:
                metrics['days_inventory_outstanding'] = 0
            
            # Days Payable Outstanding (DPO)
            if accounts_payable > 0:
                daily_cogs = revenue * 0.7 / 365
                metrics['days_payable_outstanding'] = accounts_payable / daily_cogs
            else:
                metrics['days_payable_outstanding'] = 0
            
            # Cash Conversion Cycle
            metrics['cash_conversion_cycle'] = (metrics['days_sales_outstanding'] + 
                                              metrics['days_inventory_outstanding'] - 
                                              metrics['days_payable_outstanding'])
        
        return metrics
    
    @staticmethod
    def calculate_terminal_value(final_year_fcf: float, growth_rate: float, 
                               discount_rate: float, method: str = 'gordon') -> float:
        """
        Calculate terminal value using different methods
        
        Args:
            final_year_fcf: Final year free cash flow
            growth_rate: Terminal growth rate
            discount_rate: Discount rate
            method: 'gordon' for Gordon Growth Model, 'exit_multiple' for multiple-based
            
        Returns:
            Terminal value
        """
        if method == 'gordon':
            if discount_rate <= growth_rate:
                raise ValueError("Discount rate must be greater than growth rate")
            
            terminal_fcf = final_year_fcf * (1 + growth_rate)
            return terminal_fcf / (discount_rate - growth_rate)
        
        else:
            raise ValueError("Unsupported terminal value method")
    
    @staticmethod
    def monte_carlo_dcf(base_revenue: float, scenarios: int = 1000) -> Dict[str, List[float]]:
        """
        Monte Carlo simulation for DCF analysis
        
        Args:
            base_revenue: Base year revenue
            scenarios: Number of scenarios to simulate
            
        Returns:
            Dictionary with simulation results
        """
        results = {
            'enterprise_values': [],
            'revenue_growth_rates': [],
            'ebitda_margins': [],
            'terminal_growth_rates': []
        }
        
        for _ in range(scenarios):
            # Random variables with realistic ranges
            revenue_growth = np.random.normal(0.05, 0.03)  # 5% ± 3%
            ebitda_margin = np.random.normal(0.20, 0.05)   # 20% ± 5%
            terminal_growth = np.random.normal(0.025, 0.005)  # 2.5% ± 0.5%
            wacc = np.random.normal(0.08, 0.015)           # 8% ± 1.5%
            
            # Ensure reasonable bounds
            revenue_growth = max(min(revenue_growth, 0.15), -0.05)  # -5% to 15%
            ebitda_margin = max(min(ebitda_margin, 0.40), 0.05)    # 5% to 40%
            terminal_growth = max(min(terminal_growth, 0.04), 0.01)  # 1% to 4%
            wacc = max(min(wacc, 0.15), 0.05)                     # 5% to 15%
            
            # Simple DCF calculation
            projection_years = 5
            revenues = [base_revenue * (1 + revenue_growth) ** i for i in range(1, projection_years + 1)]
            fcfs = [rev * ebitda_margin * 0.8 for rev in revenues]  # Simplified FCF
            
            # Calculate PV of FCFs
            pv_fcfs = [fcf / (1 + wacc) ** i for i, fcf in enumerate(fcfs, 1)]
            
            # Terminal value
            terminal_fcf = fcfs[-1] * (1 + terminal_growth)
            terminal_value = terminal_fcf / (wacc - terminal_growth)
            pv_terminal = terminal_value / (1 + wacc) ** projection_years
            
            enterprise_value = sum(pv_fcfs) + pv_terminal
            
            # Store results
            results['enterprise_values'].append(enterprise_value)
            results['revenue_growth_rates'].append(revenue_growth)
            results['ebitda_margins'].append(ebitda_margin)
            results['terminal_growth_rates'].append(terminal_growth)
        
        return results
    
    @staticmethod
    def calculate_option_value(underlying_price: float, strike_price: float, 
                             time_to_expiry: float, volatility: float, 
                             risk_free_rate: float, option_type: str = 'call') -> Dict[str, float]:
        """
        Calculate option value using Black-Scholes model
        
        Args:
            underlying_price: Current price of underlying asset
            strike_price: Strike price
            time_to_expiry: Time to expiry in years
            volatility: Volatility (annual)
            risk_free_rate: Risk-free rate (annual)
            option_type: 'call' or 'put'
            
        Returns:
            Dictionary with option value and Greeks
        """
        try:
            from scipy.stats import norm
            import math
            
            # Black-Scholes parameters
            d1 = (math.log(underlying_price / strike_price) + 
                  (risk_free_rate + 0.5 * volatility**2) * time_to_expiry) / (volatility * math.sqrt(time_to_expiry))
            
            d2 = d1 - volatility * math.sqrt(time_to_expiry)
            
            if option_type == 'call':
                option_value = (underlying_price * norm.cdf(d1) - 
                               strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
                delta = norm.cdf(d1)
            else:  # put
                option_value = (strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2) - 
                               underlying_price * norm.cdf(-d1))
                delta = -norm.cdf(-d1)
            
            # Greeks
            gamma = norm.pdf(d1) / (underlying_price * volatility * math.sqrt(time_to_expiry))
            theta = (-(underlying_price * norm.pdf(d1) * volatility) / (2 * math.sqrt(time_to_expiry)) - 
                    risk_free_rate * strike_price * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2 if option_type == 'call' else -d2))
            vega = underlying_price * norm.pdf(d1) * math.sqrt(time_to_expiry)
            rho = strike_price * time_to_expiry * math.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2 if option_type == 'call' else -d2)
            
            return {
                'option_value': option_value,
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho,
                'd1': d1,
                'd2': d2
            }
            
        except ImportError:
            # Fallback calculation without scipy
            intrinsic_value = max(0, underlying_price - strike_price if option_type == 'call' 
                                 else strike_price - underlying_price)
            time_value = max(0, underlying_price * 0.1 * math.sqrt(time_to_expiry))  # Rough approximation
            
            return {
                'option_value': intrinsic_value + time_value,
                'delta': 0.5,  # Default values
                'gamma': 0.01,
                'theta': -0.05,
                'vega': 0.1,
                'rho': 0.05,
                'd1': 0,
                'd2': 0
            }
