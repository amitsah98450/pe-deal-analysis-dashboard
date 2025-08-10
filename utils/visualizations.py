"""
Visualization utilities for Private Equity Dashboard
"""

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional


def create_waterfall_chart(data: List[Tuple[str, float]], title: str = "Value Creation Waterfall") -> go.Figure:
    """
    Create a waterfall chart for value creation analysis
    
    Args:
        data: List of tuples (label, value)
        title: Chart title
        
    Returns:
        Plotly figure
    """
    labels, values = zip(*data)
    
    # Calculate cumulative values for positioning
    cumulative = []
    current = 0
    
    for i, value in enumerate(values):
        if i == 0:  # First bar starts from 0
            cumulative.append(0)
            current = value
        elif i == len(values) - 1:  # Last bar (total) starts from 0
            cumulative.append(0)
        else:
            cumulative.append(current)
            current += value
    
    # Create colors (red for negative, green for positive, blue for total)
    colors = []
    for i, value in enumerate(values):
        if i == 0 or i == len(values) - 1:
            colors.append('#2E86AB')  # Blue for start and end
        elif value >= 0:
            colors.append('#A23B72')  # Green for positive
        else:
            colors.append('#F18F01')  # Red for negative
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        x=labels,
        y=values,
        base=cumulative,
        marker_color=colors,
        text=[f"${v:.0f}M" for v in values],
        textposition='outside',
        name="Value"
    ))
    
    # Add connecting lines
    for i in range(len(values) - 1):
        if i < len(values) - 2:  # Don't connect to the total bar
            start_y = cumulative[i] + values[i]
            end_y = cumulative[i + 1]
            
            fig.add_shape(
                type="line",
                x0=i + 0.4, y0=start_y,
                x1=i + 0.6, y1=end_y,
                line=dict(color="gray", dash="dot", width=1)
            )
    
    fig.update_layout(
        title=title,
        xaxis_title="Components",
        yaxis_title="Value ($M)",
        showlegend=False,
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_sensitivity_heatmap(sensitivity_df: pd.DataFrame, 
                              title: str = "Sensitivity Analysis") -> go.Figure:
    """
    Create a sensitivity heatmap
    
    Args:
        sensitivity_df: DataFrame with sensitivity data
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    fig.add_trace(go.Heatmap(
        z=sensitivity_df.values,
        x=sensitivity_df.columns,
        y=sensitivity_df.index,
        colorscale='RdYlGn',
        text=[[f"${val:.0f}M" for val in row] for row in sensitivity_df.values],
        texttemplate="%{text}",
        textfont={"size": 10},
        hoverongaps=False
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="WACC",
        yaxis_title="Revenue Growth",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_dcf_components_chart(dcf_data: Dict) -> go.Figure:
    """
    Create a chart showing DCF components
    
    Args:
        dcf_data: Dictionary with DCF calculation results
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Revenue Projections', 'Free Cash Flow', 
                       'Present Value of FCF', 'Enterprise Value Components'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"type": "pie"}]]
    )
    
    years = list(range(1, len(dcf_data['revenue_projections']) + 1))
    
    # Revenue projections
    fig.add_trace(
        go.Bar(x=years, y=dcf_data['revenue_projections'], 
               name="Revenue", marker_color='lightblue'),
        row=1, col=1
    )
    
    # Free cash flow
    fig.add_trace(
        go.Bar(x=years, y=dcf_data['fcf_projections'], 
               name="FCF", marker_color='lightgreen'),
        row=1, col=2
    )
    
    # Present value of FCF
    fig.add_trace(
        go.Bar(x=years, y=dcf_data['pv_fcfs'], 
               name="PV of FCF", marker_color='orange'),
        row=2, col=1
    )
    
    # Enterprise value pie chart
    ev_components = ['PV of FCFs', 'PV of Terminal Value']
    ev_values = [dcf_data['total_pv_fcf'], dcf_data['pv_terminal']]
    
    fig.add_trace(
        go.Pie(labels=ev_components, values=ev_values, name="EV Components"),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="DCF Model Components Analysis",
        template="plotly_white"
    )
    
    return fig


def create_lbo_returns_chart(scenarios: List[Dict], base_case: Dict) -> go.Figure:
    """
    Create LBO returns scenario analysis chart
    
    Args:
        scenarios: List of scenario dictionaries
        base_case: Base case scenario
        
    Returns:
        Plotly figure
    """
    scenario_names = ['Base Case'] + [f'Scenario {i+1}' for i in range(len(scenarios))]
    irrs = [base_case['irr']] + [s['irr'] for s in scenarios]
    multiples = [base_case['total_return']] + [s['total_return'] for s in scenarios]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('IRR by Scenario', 'Total Return Multiple'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # IRR chart
    colors = ['green' if irr >= 20 else 'orange' if irr >= 15 else 'red' for irr in irrs]
    
    fig.add_trace(
        go.Bar(x=scenario_names, y=irrs, name="IRR (%)", 
               marker_color=colors, text=[f"{irr:.1f}%" for irr in irrs]),
        row=1, col=1
    )
    
    # Multiple chart
    colors = ['green' if mult >= 2.0 else 'orange' if mult >= 1.5 else 'red' for mult in multiples]
    
    fig.add_trace(
        go.Bar(x=scenario_names, y=multiples, name="Multiple (x)", 
               marker_color=colors, text=[f"{mult:.1f}x" for mult in multiples]),
        row=1, col=2
    )
    
    fig.update_layout(
        height=500,
        showlegend=False,
        title_text="LBO Returns Analysis",
        template="plotly_white"
    )
    
    return fig


def create_debt_schedule_chart(debt_schedule: List[float], ebitda_projections: List[float]) -> go.Figure:
    """
    Create debt schedule and leverage chart
    
    Args:
        debt_schedule: List of debt balances
        ebitda_projections: List of EBITDA projections
        
    Returns:
        Plotly figure
    """
    years = list(range(len(debt_schedule)))
    leverage_ratios = [debt / ebitda if ebitda > 0 else 0 
                      for debt, ebitda in zip(debt_schedule, ebitda_projections)]
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Debt balance
    fig.add_trace(
        go.Bar(x=years, y=debt_schedule, name="Debt Balance ($M)", 
               marker_color='lightcoral'),
        secondary_y=False
    )
    
    # Leverage ratio
    fig.add_trace(
        go.Scatter(x=years, y=leverage_ratios, mode='lines+markers',
                  name="Debt/EBITDA", line=dict(color='darkblue', width=3)),
        secondary_y=True
    )
    
    # Add leverage covenant line
    fig.add_hline(y=6.0, line_dash="dash", line_color="red", 
                 annotation_text="Leverage Covenant (6.0x)", secondary_y=True)
    
    fig.update_xaxes(title_text="Year")
    fig.update_yaxes(title_text="Debt Balance ($M)", secondary_y=False)
    fig.update_yaxes(title_text="Debt/EBITDA Ratio", secondary_y=True)
    
    fig.update_layout(
        title_text="Debt Paydown Schedule and Leverage",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_comparable_multiples_chart(comp_df: pd.DataFrame, target_ticker: str) -> go.Figure:
    """
    Create comparable multiples visualization
    
    Args:
        comp_df: DataFrame with comparable analysis
        target_ticker: Target company ticker
        
    Returns:
        Plotly figure
    """
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('P/E Ratio', 'EV/EBITDA', 'EV/Revenue', 'ROE (%)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    companies = comp_df['Company'].tolist()
    colors = ['red' if comp == target_ticker else 'lightblue' for comp in companies]
    
    # P/E Ratio
    fig.add_trace(
        go.Bar(x=companies, y=comp_df['P/E Ratio'], name="P/E", 
               marker_color=colors, showlegend=False),
        row=1, col=1
    )
    
    # EV/EBITDA
    fig.add_trace(
        go.Bar(x=companies, y=comp_df['EV/EBITDA'], name="EV/EBITDA", 
               marker_color=colors, showlegend=False),
        row=1, col=2
    )
    
    # EV/Revenue
    fig.add_trace(
        go.Bar(x=companies, y=comp_df['EV/Revenue'], name="EV/Revenue", 
               marker_color=colors, showlegend=False),
        row=2, col=1
    )
    
    # ROE
    fig.add_trace(
        go.Bar(x=companies, y=comp_df['ROE (%)'], name="ROE", 
               marker_color=colors, showlegend=False),
        row=2, col=2
    )
    
    fig.update_layout(
        height=800,
        title_text="Comparable Company Analysis",
        template="plotly_white"
    )
    
    return fig


def create_monte_carlo_distribution(simulation_results: List[float], 
                                   title: str = "Monte Carlo Simulation Results") -> go.Figure:
    """
    Create Monte Carlo simulation distribution chart
    
    Args:
        simulation_results: List of simulation results
        title: Chart title
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Histogram
    fig.add_trace(go.Histogram(
        x=simulation_results,
        nbinsx=50,
        name="Distribution",
        marker_color='lightblue',
        opacity=0.7
    ))
    
    # Add percentile lines
    percentiles = [5, 25, 50, 75, 95]
    percentile_values = np.percentile(simulation_results, percentiles)
    colors = ['red', 'orange', 'green', 'orange', 'red']
    
    for p, val, color in zip(percentiles, percentile_values, colors):
        fig.add_vline(x=val, line_dash="dash", line_color=color,
                     annotation_text=f"{p}th: ${val:.0f}M")
    
    fig.update_layout(
        title=title,
        xaxis_title="Enterprise Value ($M)",
        yaxis_title="Frequency",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_risk_return_scatter(companies_data: List[Dict]) -> go.Figure:
    """
    Create risk-return scatter plot
    
    Args:
        companies_data: List of dictionaries with company risk/return data
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    for company in companies_data:
        fig.add_trace(go.Scatter(
            x=[company['risk']],
            y=[company['return']],
            mode='markers',
            marker=dict(size=company.get('size', 10)),
            name=company['name'],
            text=company['name'],
            textposition="top center"
        ))
    
    fig.update_layout(
        title="Risk-Return Analysis",
        xaxis_title="Risk (Standard Deviation)",
        yaxis_title="Expected Return (%)",
        height=500,
        template="plotly_white"
    )
    
    return fig


def create_sector_performance_radar(sector_data: pd.DataFrame) -> go.Figure:
    """
    Create radar chart for sector performance
    
    Args:
        sector_data: DataFrame with sector performance metrics
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    categories = ['Total Return', 'Sharpe Ratio', 'Max Drawdown (Inv)', 'Volatility (Inv)']
    
    for _, row in sector_data.iterrows():
        # Normalize metrics (invert negative ones)
        values = [
            row['Total_Return_%'],
            row['Sharpe_Ratio'] * 10,  # Scale up for visibility
            -row['Max_Drawdown_%'],  # Invert (less negative is better)
            -row['Volatility_%']  # Invert (less volatile is better)
        ]
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=row['Sector']
        ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[-50, 50]
            )),
        showlegend=True,
        title="Sector Performance Comparison",
        height=600
    )
    
    return fig
