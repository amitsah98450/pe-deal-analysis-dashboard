import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# Import custom modules
from models.dcf_model import DCFModel
from models.lbo_model import LBOModel
from models.valuation import ValuationModel
from data.data_fetcher import DataFetcher
from utils.visualizations import create_waterfall_chart, create_sensitivity_heatmap
from config.settings import APP_CONFIG

# Page configuration
st.set_page_config(
    page_title="PE Deal Analysis Dashboard",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 0.5rem 0;
    }
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """Main application function"""
    
    # Header
    st.markdown('<h1 class="main-header">🏦 Private Equity Deal Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar for navigation and inputs
    with st.sidebar:
        st.markdown('<div class="sidebar-header">📊 Analysis Options</div>', 
                    unsafe_allow_html=True)
        
        # Navigation
        analysis_type = st.selectbox(
            "Select Analysis Type",
            ["Deal Overview", "DCF Valuation", "LBO Model", "Comparable Analysis", 
             "Industry Benchmarks", "Risk Assessment"]
        )
        
        st.divider()
        
        # Company input
        st.subheader("Target Company")
        ticker = st.text_input("Stock Ticker", value="AAPL", help="Enter stock ticker symbol")
        
        # Financial inputs
        st.subheader("Key Assumptions")
        revenue_growth = st.slider("Revenue Growth (%)", 0.0, 50.0, 5.0, 0.5)
        ebitda_margin = st.slider("EBITDA Margin (%)", 5.0, 50.0, 20.0, 1.0)
        terminal_growth = st.slider("Terminal Growth (%)", 1.0, 5.0, 2.5, 0.1)
        wacc = st.slider("WACC (%)", 5.0, 15.0, 8.5, 0.1)
        
        st.divider()
        
        # Load data button
        if st.button("🔄 Refresh Data", type="primary"):
            st.rerun()
    
    # Main content area
    if analysis_type == "Deal Overview":
        show_deal_overview(ticker)
    elif analysis_type == "DCF Valuation":
        show_dcf_analysis(ticker, revenue_growth, ebitda_margin, terminal_growth, wacc)
    elif analysis_type == "LBO Model":
        show_lbo_analysis(ticker, revenue_growth, ebitda_margin)
    elif analysis_type == "Comparable Analysis":
        show_comparable_analysis(ticker)
    elif analysis_type == "Industry Benchmarks":
        show_industry_benchmarks(ticker)
    elif analysis_type == "Risk Assessment":
        show_risk_assessment(ticker, revenue_growth, ebitda_margin, wacc)

def show_deal_overview(ticker: str):
    """Display deal overview and key metrics"""
    st.header("📋 Deal Overview")
    
    try:
        # Fetch company data with caching
        @st.cache_data(ttl=300)  # Cache for 5 minutes
        def get_cached_company_data(ticker):
            data_fetcher = DataFetcher()
            return data_fetcher.get_company_info(ticker), data_fetcher.get_stock_data(ticker, period="1y")
        
        company_data, stock_data = get_cached_company_data(ticker)
        
        if company_data and not stock_data.empty:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Company", company_data.get('shortName', ticker))
                st.metric("Sector", company_data.get('sector', 'N/A'))
                st.metric("Market Cap", f"${company_data.get('marketCap', 0) / 1e9:.1f}B")
            
            with col2:
                current_price = stock_data['Close'].iloc[-1]
                # Calculate price change safely - use available data or default to 0
                if len(stock_data) >= 252:
                    year_ago_price = stock_data['Close'].iloc[-252]
                elif len(stock_data) >= 2:
                    year_ago_price = stock_data['Close'].iloc[0]  # Use first available price
                else:
                    year_ago_price = current_price  # No change if only one data point
                
                price_change = ((current_price - year_ago_price) / year_ago_price) * 100 if year_ago_price != 0 else 0
                st.metric("Current Price", f"${current_price:.2f}")
                st.metric("52W Return", f"{price_change:.1f}%")
                st.metric("Enterprise Value", f"${company_data.get('enterpriseValue', 0) / 1e9:.1f}B")
            
            with col3:
                st.metric("Revenue TTM", f"${company_data.get('totalRevenue', 0) / 1e9:.1f}B")
                st.metric("EBITDA TTM", f"${company_data.get('ebitda', 0) / 1e9:.1f}B")
                st.metric("P/E Ratio", f"{company_data.get('trailingPE', 0):.1f}")
            
            # Stock price chart
            st.subheader("📈 Stock Performance")
            fig = px.line(stock_data.reset_index(), x='Date', y='Close', 
                         title=f"{ticker} Stock Price (1 Year)")
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Financial summary
            st.subheader("💰 Financial Summary")
            financial_data = {
                'Metric': ['Total Revenue', 'Gross Profit', 'EBITDA', 'Net Income', 'Total Debt'],
                'Value ($B)': [
                    company_data.get('totalRevenue', 0) / 1e9,
                    company_data.get('grossProfits', 0) / 1e9,
                    company_data.get('ebitda', 0) / 1e9,
                    company_data.get('netIncomeToCommon', 0) / 1e9,
                    company_data.get('totalDebt', 0) / 1e9
                ]
            }
            
            financial_df = pd.DataFrame(financial_data)
            financial_df['Value ($B)'] = financial_df['Value ($B)'].round(2)
            st.dataframe(financial_df, use_container_width=True)
        
        else:
            st.error(f"Unable to fetch data for ticker: {ticker}")
            st.info("Please verify the ticker symbol is correct and the company is publicly traded.")
    
    except Exception as e:
        st.error(f"Error loading data for {ticker}: {str(e)}")
        st.info("Please check the ticker symbol and try again. Some companies may have limited data availability.")

def show_dcf_analysis(ticker: str, revenue_growth: float, ebitda_margin: float, 
                     terminal_growth: float, wacc: float):
    """Display DCF valuation analysis"""
    st.header("📊 DCF Valuation Model")
    
    # Initialize DCF model
    dcf_model = DCFModel()
    data_fetcher = DataFetcher()
    company_data = data_fetcher.get_company_info(ticker)
    
    if company_data:
        # Get base revenue
        base_revenue = company_data.get('totalRevenue', 1e9) / 1e6  # Convert to millions
        
        # Calculate DCF
        dcf_result = dcf_model.calculate_dcf(
            base_revenue=base_revenue,
            revenue_growth=revenue_growth / 100,
            ebitda_margin=ebitda_margin / 100,
            terminal_growth=terminal_growth / 100,
            wacc=wacc / 100
        )
        
        # Display results
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Enterprise Value", f"${dcf_result['enterprise_value']:.0f}M")
        with col2:
            st.metric("Equity Value", f"${dcf_result['equity_value']:.0f}M")
        with col3:
            st.metric("Value per Share", f"${dcf_result['value_per_share']:.2f}")
        with col4:
            current_price = data_fetcher.get_current_price(ticker)
            upside = ((dcf_result['value_per_share'] - current_price) / current_price) * 100 if current_price else 0
            st.metric("Upside/Downside", f"{upside:.1f}%")
        
        # Cash flow projections
        st.subheader("💰 Free Cash Flow Projections")
        fig = px.bar(x=list(range(1, 6)), y=dcf_result['fcf_projections'],
                    title="5-Year Free Cash Flow Projections",
                    labels={'x': 'Year', 'y': 'Free Cash Flow ($M)'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Sensitivity analysis
        st.subheader("🌡️ Sensitivity Analysis")
        sensitivity_data = dcf_model.sensitivity_analysis(
            base_revenue, ebitda_margin / 100, terminal_growth / 100, wacc / 100
        )
        
        fig_heatmap = create_sensitivity_heatmap(sensitivity_data)
        st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # DCF assumptions table
        st.subheader("📋 Model Assumptions")
        assumptions_df = pd.DataFrame({
            'Parameter': ['Revenue Growth', 'EBITDA Margin', 'Terminal Growth', 'WACC', 'Tax Rate'],
            'Value': [f"{revenue_growth}%", f"{ebitda_margin}%", f"{terminal_growth}%", 
                     f"{wacc}%", "25%"]
        })
        st.dataframe(assumptions_df, use_container_width=True)

def show_lbo_analysis(ticker: str, revenue_growth: float, ebitda_margin: float):
    """Display LBO model analysis"""
    st.header("🏗️ LBO Model Analysis")
    
    # LBO inputs
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Transaction Assumptions")
        purchase_multiple = st.slider("Purchase EV/EBITDA", 8.0, 20.0, 12.0, 0.5)
        exit_multiple = st.slider("Exit EV/EBITDA", 8.0, 20.0, 10.0, 0.5)
        debt_multiple = st.slider("Debt/EBITDA", 3.0, 8.0, 5.0, 0.5)
        
    with col2:
        st.subheader("⚙️ Deal Structure")
        management_rollover = st.slider("Management Rollover (%)", 0.0, 20.0, 5.0, 1.0)
        transaction_fees = st.slider("Transaction Fees (%)", 1.0, 5.0, 2.5, 0.5)
        hold_period = st.slider("Hold Period (years)", 3, 7, 5)
    
    # Initialize LBO model
    lbo_model = LBOModel()
    data_fetcher = DataFetcher()
    company_data = data_fetcher.get_company_info(ticker)
    
    if company_data:
        base_ebitda = company_data.get('ebitda', 1e8) / 1e6  # Convert to millions
        
        lbo_result = lbo_model.calculate_lbo_returns(
            ebitda=base_ebitda,
            purchase_multiple=purchase_multiple,
            exit_multiple=exit_multiple,
            debt_multiple=debt_multiple,
            revenue_growth=revenue_growth / 100,
            ebitda_margin=ebitda_margin / 100,
            hold_period=hold_period
        )
        
        # Display key metrics
        st.subheader("🎯 LBO Returns Analysis")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Equity Investment", f"${lbo_result['equity_investment']:.0f}M")
        with col2:
            st.metric("Exit Equity Value", f"${lbo_result['exit_equity_value']:.0f}M")
        with col3:
            st.metric("Total Return", f"{lbo_result['total_return']:.1f}x")
        with col4:
            st.metric("IRR", f"{lbo_result['irr']:.1f}%")
        
        # Cash flow waterfall
        st.subheader("💧 Value Creation Waterfall")
        waterfall_data = [
            ("Purchase Price", -lbo_result['purchase_price']),
            ("EBITDA Growth", lbo_result['ebitda_growth_value']),
            ("Multiple Expansion", lbo_result['multiple_expansion']),
            ("Debt Paydown", lbo_result['debt_paydown']),
            ("Exit Value", lbo_result['exit_equity_value'])
        ]
        
        fig_waterfall = create_waterfall_chart(waterfall_data)
        st.plotly_chart(fig_waterfall, use_container_width=True)
        
        # Debt schedule
        st.subheader("📈 Debt Paydown Schedule")
        debt_schedule = lbo_model.create_debt_schedule(
            initial_debt=lbo_result['initial_debt'],
            base_ebitda=base_ebitda,
            revenue_growth=revenue_growth / 100,
            years=hold_period
        )
        
        fig_debt = px.line(x=list(range(hold_period + 1)), y=debt_schedule,
                          title="Debt Outstanding Over Time",
                          labels={'x': 'Year', 'y': 'Debt Outstanding ($M)'})
        st.plotly_chart(fig_debt, use_container_width=True)

def show_comparable_analysis(ticker: str):
    """Display comparable company analysis"""
    st.header("🔍 Comparable Company Analysis")
    
    # Get peer companies (simplified approach using sector data)
    data_fetcher = DataFetcher()
    company_data = data_fetcher.get_company_info(ticker)
    
    if company_data:
        sector = company_data.get('sector', 'Technology')
        
        # Sample peer companies by sector (in a real application, you'd have a comprehensive database)
        sector_peers = {
            'Technology': ['AAPL', 'MSFT', 'GOOGL', 'META', 'NFLX'],
            'Healthcare': ['JNJ', 'PFE', 'UNH', 'ABBV', 'TMO'],
            'Financial Services': ['JPM', 'BAC', 'WFC', 'C', 'GS'],
            'Consumer Cyclical': ['AMZN', 'TSLA', 'HD', 'MCD', 'NKE'],
            'Energy': ['XOM', 'CVX', 'COP', 'EOG', 'SLB']
        }
        
        peers = sector_peers.get(sector, ['AAPL', 'MSFT', 'GOOGL'])[:5]
        
        # Create comparable analysis
        valuation_model = ValuationModel()
        comp_analysis = valuation_model.comparable_analysis(ticker, peers)
        
        if comp_analysis is not None and not comp_analysis.empty:
            st.subheader("📊 Trading Multiples Comparison")
            
            # Display multiples table
            st.dataframe(comp_analysis, use_container_width=True)
            
            # Visualize multiples
            col1, col2 = st.columns(2)
            
            with col1:
                fig_pe = px.bar(comp_analysis, x='Company', y='P/E Ratio',
                               title="P/E Ratio Comparison")
                st.plotly_chart(fig_pe, use_container_width=True)
            
            with col2:
                fig_ev = px.bar(comp_analysis, x='Company', y='EV/EBITDA',
                               title="EV/EBITDA Comparison")
                st.plotly_chart(fig_ev, use_container_width=True)
            
            # Summary statistics
            st.subheader("📈 Peer Group Statistics")
            summary_stats = comp_analysis.describe()
            st.dataframe(summary_stats, use_container_width=True)

def show_industry_benchmarks(ticker: str):
    """Display industry benchmarking analysis"""
    st.header("🏭 Industry Benchmarks")
    
    data_fetcher = DataFetcher()
    company_data = data_fetcher.get_company_info(ticker)
    
    if company_data:
        sector = company_data.get('sector', 'Technology')
        
        # Industry benchmark data (simplified)
        benchmark_data = {
            'Technology': {
                'Median P/E': 25.5, 'Median EV/EBITDA': 18.2, 'Median ROE': 22.5,
                'Revenue Growth': 12.5, 'EBITDA Margin': 28.5, 'Debt/Equity': 0.35
            },
            'Healthcare': {
                'Median P/E': 18.7, 'Median EV/EBITDA': 14.8, 'Median ROE': 18.2,
                'Revenue Growth': 8.5, 'EBITDA Margin': 24.2, 'Debt/Equity': 0.42
            },
            'Financial Services': {
                'Median P/E': 12.3, 'Median EV/EBITDA': 8.5, 'Median ROE': 11.8,
                'Revenue Growth': 6.2, 'EBITDA Margin': 35.8, 'Debt/Equity': 2.15
            }
        }
        
        industry_metrics = benchmark_data.get(sector, benchmark_data['Technology'])
        
        # Current company metrics
        current_pe = company_data.get('trailingPE', 0)
        current_roe = company_data.get('returnOnEquity', 0) * 100 if company_data.get('returnOnEquity') else 0
        
        st.subheader(f"📊 {sector} Industry Benchmarks")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Industry Median P/E", f"{industry_metrics['Median P/E']:.1f}")
            st.metric("Company P/E", f"{current_pe:.1f}")
            
        with col2:
            st.metric("Industry EV/EBITDA", f"{industry_metrics['Median EV/EBITDA']:.1f}")
            st.metric("Industry Revenue Growth", f"{industry_metrics['Revenue Growth']:.1f}%")
            
        with col3:
            st.metric("Industry ROE", f"{industry_metrics['Median ROE']:.1f}%")
            st.metric("Company ROE", f"{current_roe:.1f}%")
        
        # Benchmark comparison chart
        metrics = ['P/E Ratio', 'EV/EBITDA', 'ROE', 'Revenue Growth']
        company_values = [current_pe, 15.5, current_roe, 8.2]  # Sample values
        industry_values = [industry_metrics['Median P/E'], industry_metrics['Median EV/EBITDA'],
                          industry_metrics['Median ROE'], industry_metrics['Revenue Growth']]
        
        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=company_values,
            theta=metrics,
            fill='toself',
            name=f'{ticker} Company'
        ))
        fig.add_trace(go.Scatterpolar(
            r=industry_values,
            theta=metrics,
            fill='toself',
            name='Industry Median'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, max(max(company_values), max(industry_values)) * 1.2]
                )),
            showlegend=True,
            title="Company vs Industry Benchmark Comparison"
        )
        
        st.plotly_chart(fig, use_container_width=True)

def show_risk_assessment(ticker: str, revenue_growth: float, ebitda_margin: float, wacc: float):
    """Display risk assessment and Monte Carlo simulation"""
    st.header("⚠️ Risk Assessment")
    
    # Risk parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎲 Monte Carlo Parameters")
        revenue_volatility = st.slider("Revenue Growth Volatility (%)", 1.0, 10.0, 3.0, 0.5)
        margin_volatility = st.slider("EBITDA Margin Volatility (%)", 1.0, 5.0, 2.0, 0.5)
        simulations = st.slider("Number of Simulations", 1000, 10000, 5000, 1000)
    
    with col2:
        st.subheader("📊 Risk Factors")
        market_risk = st.slider("Market Risk Weight", 0.1, 1.0, 0.7, 0.1)
        operational_risk = st.slider("Operational Risk Weight", 0.1, 1.0, 0.5, 0.1)
        financial_risk = st.slider("Financial Risk Weight", 0.1, 1.0, 0.6, 0.1)
    
    # Run Monte Carlo simulation
    data_fetcher = DataFetcher()
    company_data = data_fetcher.get_company_info(ticker)
    
    if company_data:
        base_revenue = company_data.get('totalRevenue', 1e9) / 1e6
        
        # Monte Carlo simulation
        dcf_model = DCFModel()
        mc_results = dcf_model.monte_carlo_simulation(
            base_revenue=base_revenue,
            base_growth=revenue_growth / 100,
            base_margin=ebitda_margin / 100,
            wacc=wacc / 100,
            revenue_volatility=revenue_volatility / 100,
            margin_volatility=margin_volatility / 100,
            simulations=simulations
        )
        
        # Display results
        st.subheader("🎯 Valuation Distribution")
        
        # Convert to numpy array for consistent calculations
        mc_array = np.array(mc_results)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Mean Valuation", f"${np.mean(mc_array):.0f}M")
        with col2:
            st.metric("Median Valuation", f"${np.median(mc_array):.0f}M")
        with col3:
            st.metric("5th Percentile", f"${np.percentile(mc_array, 5):.0f}M")
        with col4:
            st.metric("95th Percentile", f"${np.percentile(mc_array, 95):.0f}M")
        
        # Histogram of results
        fig_hist = px.histogram(x=mc_array, nbins=50,
                               title="Monte Carlo Valuation Distribution",
                               labels={'x': 'Enterprise Value ($M)', 'y': 'Frequency'})
        st.plotly_chart(fig_hist, use_container_width=True)
        
        # Risk metrics
        st.subheader("📊 Risk Metrics")
        
        var_95 = np.percentile(mc_array, 5)
        var_99 = np.percentile(mc_array, 1)
        mean_val = np.mean(mc_array)
        
        # Calculate Expected Shortfall safely
        try:
            tail_mask = mc_array <= var_95
            tail_losses = mc_array[tail_mask]
            expected_shortfall = np.mean(tail_losses) if len(tail_losses) > 0 else var_95
        except Exception as e:
            st.warning(f"Error calculating Expected Shortfall: {e}")
            expected_shortfall = var_95
        
        risk_metrics = pd.DataFrame({
            'Risk Metric': ['Value at Risk (95%)', 'Value at Risk (99%)', 'Expected Shortfall',
                           'Volatility', 'Sharpe Ratio'],
            'Value': [
                f"${var_95:.0f}M",
                f"${var_99:.0f}M",
                f"${expected_shortfall:.0f}M",
                f"{np.std(mc_array) / mean_val * 100:.1f}%",
                f"{(mean_val - wacc * mean_val) / np.std(mc_array):.2f}"
            ]
        })
        
        st.dataframe(risk_metrics, use_container_width=True)

if __name__ == "__main__":
    main()
