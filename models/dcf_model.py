"""
DCF (Discounted Cash Flow) Model for Private Equity Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from scipy.stats import norm


class DCFModel:
    """
    Comprehensive DCF model for private equity deal analysis
    """
    
    def __init__(self):
        self.tax_rate = 0.25  # Default corporate tax rate
        self.projection_years = 5
        
    def calculate_dcf(self, base_revenue: float, revenue_growth: float, 
                     ebitda_margin: float, terminal_growth: float, 
                     wacc: float, capex_rate: float = 0.03, 
                     nwc_rate: float = 0.02) -> Dict:
        """
        Calculate DCF valuation with detailed projections
        
        Args:
            base_revenue: Base year revenue in millions
            revenue_growth: Annual revenue growth rate
            ebitda_margin: EBITDA as % of revenue
            terminal_growth: Terminal growth rate
            wacc: Weighted average cost of capital
            capex_rate: CapEx as % of revenue
            nwc_rate: Net working capital change as % of revenue
            
        Returns:
            Dictionary with DCF results
        """
        
        # Revenue projections
        revenues = [base_revenue * (1 + revenue_growth) ** i for i in range(1, self.projection_years + 1)]
        
        # EBITDA projections
        ebitdas = [revenue * ebitda_margin for revenue in revenues]
        
        # EBIT (assuming D&A is 2% of revenue)
        da_rate = 0.02
        ebits = [ebitda - revenue * da_rate for ebitda, revenue in zip(ebitdas, revenues)]
        
        # Tax calculations
        taxes = [ebit * self.tax_rate for ebit in ebits]
        
        # NOPAT (Net Operating Profit After Tax)
        nopats = [ebit - tax for ebit, tax in zip(ebits, taxes)]
        
        # Add back D&A
        da_amounts = [revenue * da_rate for revenue in revenues]
        
        # CapEx
        capex_amounts = [revenue * capex_rate for revenue in revenues]
        
        # Net Working Capital changes
        nwc_changes = [revenue * nwc_rate for revenue in revenues]
        
        # Free Cash Flow calculations
        fcfs = []
        for i in range(self.projection_years):
            fcf = nopats[i] + da_amounts[i] - capex_amounts[i] - nwc_changes[i]
            fcfs.append(fcf)
        
        # Terminal value calculation
        terminal_fcf = fcfs[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcf / (wacc - terminal_growth)
        
        # Discount factors
        discount_factors = [(1 + wacc) ** -i for i in range(1, self.projection_years + 1)]
        terminal_discount_factor = (1 + wacc) ** -self.projection_years
        
        # Present values
        pv_fcfs = [fcf * df for fcf, df in zip(fcfs, discount_factors)]
        pv_terminal = terminal_value * terminal_discount_factor
        
        # Enterprise value
        enterprise_value = sum(pv_fcfs) + pv_terminal
        
        # Assuming net debt of 10% of enterprise value for equity value calculation
        net_debt = enterprise_value * 0.1
        equity_value = enterprise_value - net_debt
        
        # Assuming 100M shares outstanding
        shares_outstanding = 100
        value_per_share = equity_value / shares_outstanding
        
        return {
            'enterprise_value': enterprise_value,
            'equity_value': equity_value,
            'value_per_share': value_per_share,
            'terminal_value': terminal_value,
            'pv_terminal': pv_terminal,
            'fcf_projections': fcfs,
            'revenue_projections': revenues,
            'ebitda_projections': ebitdas,
            'pv_fcfs': pv_fcfs,
            'total_pv_fcf': sum(pv_fcfs)
        }
    
    def sensitivity_analysis(self, base_revenue: float, ebitda_margin: float, 
                           terminal_growth: float, wacc: float) -> pd.DataFrame:
        """
        Perform sensitivity analysis on DCF valuation
        
        Returns:
            DataFrame with sensitivity results
        """
        
        # Define ranges for sensitivity
        revenue_growth_range = np.arange(0.02, 0.12, 0.02)  # 2% to 12%
        wacc_range = np.arange(0.06, 0.12, 0.01)  # 6% to 12%
        
        sensitivity_matrix = []
        
        for growth in revenue_growth_range:
            row = []
            for wacc_val in wacc_range:
                dcf_result = self.calculate_dcf(
                    base_revenue=base_revenue,
                    revenue_growth=growth,
                    ebitda_margin=ebitda_margin,
                    terminal_growth=terminal_growth,
                    wacc=wacc_val
                )
                row.append(dcf_result['enterprise_value'])
            sensitivity_matrix.append(row)
        
        sensitivity_df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{g:.1%}" for g in revenue_growth_range],
            columns=[f"{w:.1%}" for w in wacc_range]
        )
        
        return sensitivity_df
    
    def monte_carlo_simulation(self, base_revenue: float, base_growth: float,
                             base_margin: float, wacc: float, 
                             revenue_volatility: float = 0.03,
                             margin_volatility: float = 0.02,
                             simulations: int = 5000) -> List[float]:
        """
        Run Monte Carlo simulation for DCF valuation
        
        Args:
            base_revenue: Base revenue
            base_growth: Base revenue growth rate
            base_margin: Base EBITDA margin
            wacc: Weighted average cost of capital
            revenue_volatility: Standard deviation of revenue growth
            margin_volatility: Standard deviation of EBITDA margin
            simulations: Number of simulations to run
            
        Returns:
            List of enterprise values from simulations
        """
        
        results = []
        
        for _ in range(simulations):
            # Random variables
            growth_shock = np.random.normal(0, revenue_volatility)
            margin_shock = np.random.normal(0, margin_volatility)
            
            # Adjusted parameters
            sim_growth = max(base_growth + growth_shock, 0.01)  # Minimum 1% growth
            sim_margin = max(min(base_margin + margin_shock, 0.5), 0.05)  # 5-50% range
            
            # Calculate DCF with random parameters
            dcf_result = self.calculate_dcf(
                base_revenue=base_revenue,
                revenue_growth=sim_growth,
                ebitda_margin=sim_margin,
                terminal_growth=0.025,  # Fixed terminal growth
                wacc=wacc
            )
            
            results.append(dcf_result['enterprise_value'])
        
        return results
    
    def calculate_wacc(self, risk_free_rate: float, market_premium: float,
                      beta: float, tax_rate: float, debt_ratio: float,
                      cost_of_debt: float) -> float:
        """
        Calculate Weighted Average Cost of Capital (WACC)
        
        Args:
            risk_free_rate: Risk-free rate (e.g., 10-year Treasury)
            market_premium: Equity risk premium
            beta: Company beta
            tax_rate: Corporate tax rate
            debt_ratio: Debt / (Debt + Equity)
            cost_of_debt: Pre-tax cost of debt
            
        Returns:
            WACC as decimal
        """
        
        cost_of_equity = risk_free_rate + beta * market_premium
        equity_ratio = 1 - debt_ratio
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        
        wacc = (cost_of_equity * equity_ratio) + (after_tax_cost_of_debt * debt_ratio)
        
        return wacc
    
    def calculate_terminal_value(self, final_year_fcf: float, terminal_growth: float,
                               wacc: float, exit_multiple: float = None,
                               terminal_ebitda: float = None) -> Dict[str, float]:
        """
        Calculate terminal value using both Gordon Growth and Exit Multiple methods
        
        Args:
            final_year_fcf: Free cash flow in final projection year
            terminal_growth: Terminal growth rate
            wacc: Weighted average cost of capital
            exit_multiple: Exit multiple (EV/EBITDA)
            terminal_ebitda: EBITDA in terminal year
            
        Returns:
            Dictionary with both terminal value calculations
        """
        
        # Gordon Growth Model
        terminal_fcf = final_year_fcf * (1 + terminal_growth)
        gordon_terminal_value = terminal_fcf / (wacc - terminal_growth)
        
        results = {
            'gordon_growth_terminal_value': gordon_terminal_value,
            'terminal_fcf': terminal_fcf
        }
        
        # Exit Multiple Method (if provided)
        if exit_multiple and terminal_ebitda:
            exit_terminal_value = terminal_ebitda * exit_multiple
            results['exit_multiple_terminal_value'] = exit_terminal_value
            results['blended_terminal_value'] = (gordon_terminal_value + exit_terminal_value) / 2
        
        return results
