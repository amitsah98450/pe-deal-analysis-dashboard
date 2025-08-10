"""
LBO (Leveraged Buyout) Model for Private Equity Analysis
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class LBOModel:
    """
    Comprehensive LBO model for private equity transactions
    """
    
    def __init__(self):
        self.tax_rate = 0.25
        self.cash_sweep_ratio = 0.75  # % of excess cash used for debt paydown
        
    def calculate_lbo_returns(self, ebitda: float, purchase_multiple: float,
                            exit_multiple: float, debt_multiple: float,
                            revenue_growth: float, ebitda_margin: float,
                            hold_period: int = 5, interest_rate: float = 0.06) -> Dict:
        """
        Calculate LBO returns and key metrics
        
        Args:
            ebitda: Base year EBITDA in millions
            purchase_multiple: Purchase EV/EBITDA multiple
            exit_multiple: Exit EV/EBITDA multiple
            debt_multiple: Debt/EBITDA multiple
            revenue_growth: Annual revenue growth rate
            ebitda_margin: EBITDA margin assumption
            hold_period: Investment holding period in years
            interest_rate: Interest rate on debt
            
        Returns:
            Dictionary with LBO analysis results
        """
        
        # Purchase price calculation
        purchase_price = ebitda * purchase_multiple
        initial_debt = ebitda * debt_multiple
        equity_investment = purchase_price - initial_debt
        
        # Project EBITDA growth
        exit_ebitda = ebitda * (1 + revenue_growth) ** hold_period
        
        # Calculate debt paydown
        debt_schedule = self.create_debt_schedule(
            initial_debt, ebitda, revenue_growth, hold_period, interest_rate
        )
        
        final_debt = debt_schedule[-1]
        
        # Exit calculations
        exit_enterprise_value = exit_ebitda * exit_multiple
        exit_equity_value = exit_enterprise_value - final_debt
        
        # Returns calculation
        total_return = exit_equity_value / equity_investment
        irr = (total_return ** (1 / hold_period)) - 1
        
        # Value creation components
        ebitda_growth_value = (exit_ebitda - ebitda) * exit_multiple
        multiple_expansion = exit_ebitda * (exit_multiple - purchase_multiple)
        debt_paydown_value = initial_debt - final_debt
        
        return {
            'purchase_price': purchase_price,
            'equity_investment': equity_investment,
            'initial_debt': initial_debt,
            'exit_enterprise_value': exit_enterprise_value,
            'exit_equity_value': exit_equity_value,
            'total_return': total_return,
            'irr': irr * 100,
            'ebitda_growth_value': ebitda_growth_value,
            'multiple_expansion': multiple_expansion,
            'debt_paydown': debt_paydown_value,
            'exit_ebitda': exit_ebitda,
            'final_debt': final_debt
        }
    
    def create_debt_schedule(self, initial_debt: float, base_ebitda: float,
                           revenue_growth: float, years: int, 
                           interest_rate: float = 0.06) -> List[float]:
        """
        Create debt paydown schedule based on cash generation
        
        Args:
            initial_debt: Initial debt amount
            base_ebitda: Base year EBITDA
            revenue_growth: Annual revenue growth
            years: Number of years for projection
            interest_rate: Interest rate on debt
            
        Returns:
            List of debt balances by year
        """
        
        debt_balances = [initial_debt]
        current_debt = initial_debt
        
        for year in range(1, years + 1):
            # Project EBITDA for the year
            current_ebitda = base_ebitda * (1 + revenue_growth) ** year
            
            # Calculate cash available for debt paydown
            # Simplified: EBITDA - taxes - capex - interest
            interest_expense = current_debt * interest_rate
            ebit = current_ebitda - (current_ebitda * 0.02)  # Assume D&A = 2% of EBITDA
            taxes = max(0, (ebit - interest_expense) * self.tax_rate)
            capex = current_ebitda * 0.15  # Assume capex = 15% of EBITDA
            
            cash_after_expenses = current_ebitda - interest_expense - taxes - capex
            debt_paydown = max(0, cash_after_expenses * self.cash_sweep_ratio)
            
            # Update debt balance
            current_debt = max(0, current_debt - debt_paydown)
            debt_balances.append(current_debt)
        
        return debt_balances
    
    def sensitivity_analysis(self, base_ebitda: float, purchase_multiple: float,
                           debt_multiple: float, hold_period: int) -> pd.DataFrame:
        """
        Perform sensitivity analysis on LBO returns
        
        Returns:
            DataFrame with IRR sensitivity to exit multiple and revenue growth
        """
        
        exit_multiples = np.arange(8.0, 16.0, 1.0)
        revenue_growth_rates = np.arange(0.02, 0.12, 0.02)
        
        sensitivity_matrix = []
        
        for growth in revenue_growth_rates:
            row = []
            for exit_mult in exit_multiples:
                lbo_result = self.calculate_lbo_returns(
                    ebitda=base_ebitda,
                    purchase_multiple=purchase_multiple,
                    exit_multiple=exit_mult,
                    debt_multiple=debt_multiple,
                    revenue_growth=growth,
                    ebitda_margin=0.25,  # Assumed margin
                    hold_period=hold_period
                )
                row.append(lbo_result['irr'])
            sensitivity_matrix.append(row)
        
        sensitivity_df = pd.DataFrame(
            sensitivity_matrix,
            index=[f"{g:.1%}" for g in revenue_growth_rates],
            columns=[f"{m:.1f}x" for m in exit_multiples]
        )
        
        return sensitivity_df
    
    def calculate_dividend_capacity(self, ebitda: float, debt_balance: float,
                                  capex_rate: float = 0.15, 
                                  minimum_cash: float = 50) -> float:
        """
        Calculate dividend capacity for the portfolio company
        
        Args:
            ebitda: Current year EBITDA
            debt_balance: Current debt balance
            capex_rate: Capital expenditure as % of EBITDA
            minimum_cash: Minimum cash balance to maintain
            
        Returns:
            Maximum dividend that can be paid
        """
        
        # Calculate available cash
        capex = ebitda * capex_rate
        interest_expense = debt_balance * 0.06  # Assumed 6% interest rate
        taxes = max(0, (ebitda * 0.98 - interest_expense) * self.tax_rate)  # 98% = EBIT/EBITDA
        
        available_cash = ebitda - capex - interest_expense - taxes - minimum_cash
        
        # Apply debt covenant constraints (typically 1.0x minimum debt/EBITDA)
        minimum_debt_balance = ebitda * 1.0
        debt_paydown_required = max(0, debt_balance - minimum_debt_balance)
        
        dividend_capacity = max(0, available_cash - debt_paydown_required)
        
        return dividend_capacity
    
    def stress_test_scenarios(self, base_case: Dict, stress_scenarios: List[Dict]) -> pd.DataFrame:
        """
        Run stress test scenarios on the LBO model
        
        Args:
            base_case: Base case assumptions dictionary
            stress_scenarios: List of stress scenario dictionaries
            
        Returns:
            DataFrame with stress test results
        """
        
        results = []
        
        # Base case
        base_result = self.calculate_lbo_returns(**base_case)
        results.append({
            'Scenario': 'Base Case',
            'IRR': base_result['irr'],
            'Total Return': base_result['total_return'],
            'Exit Equity Value': base_result['exit_equity_value']
        })
        
        # Stress scenarios
        for i, scenario in enumerate(stress_scenarios):
            scenario_params = {**base_case, **scenario}
            scenario_result = self.calculate_lbo_returns(**scenario_params)
            
            results.append({
                'Scenario': f'Stress {i+1}',
                'IRR': scenario_result['irr'],
                'Total Return': scenario_result['total_return'],
                'Exit Equity Value': scenario_result['exit_equity_value']
            })
        
        return pd.DataFrame(results)
    
    def calculate_credit_metrics(self, ebitda_projections: List[float], 
                               debt_schedule: List[float]) -> Dict:
        """
        Calculate key credit metrics for debt sizing
        
        Args:
            ebitda_projections: List of EBITDA projections
            debt_schedule: List of debt balances
            
        Returns:
            Dictionary with credit metrics
        """
        
        # Calculate leverage ratios
        leverage_ratios = [debt / ebitda for debt, ebitda in zip(debt_schedule, ebitda_projections)]
        
        # Interest coverage ratios (simplified)
        interest_coverage_ratios = []
        for ebitda, debt in zip(ebitda_projections, debt_schedule):
            interest_expense = debt * 0.06  # 6% interest rate
            if interest_expense > 0:
                coverage = ebitda / interest_expense
            else:
                coverage = float('inf')
            interest_coverage_ratios.append(coverage)
        
        # Debt service coverage
        debt_service_coverage = []
        for i, (ebitda, debt) in enumerate(zip(ebitda_projections, debt_schedule)):
            if i == 0:
                debt_service_coverage.append(float('inf'))  # No payment in year 0
            else:
                principal_payment = debt_schedule[i-1] - debt
                interest_payment = debt_schedule[i-1] * 0.06
                total_debt_service = principal_payment + interest_payment
                
                if total_debt_service > 0:
                    dscr = ebitda / total_debt_service
                else:
                    dscr = float('inf')
                debt_service_coverage.append(dscr)
        
        return {
            'leverage_ratios': leverage_ratios,
            'interest_coverage_ratios': interest_coverage_ratios,
            'debt_service_coverage': debt_service_coverage,
            'peak_leverage': max(leverage_ratios),
            'minimum_interest_coverage': min(interest_coverage_ratios),
            'minimum_dscr': min(debt_service_coverage[1:])  # Exclude year 0
        }
