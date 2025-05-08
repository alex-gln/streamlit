import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector as connector
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Funding Rate Comparison", layout="wide")
st.title("Crypto Funding Rate Comparison Dashboard")
st.write("Compare funding rates between exchanges for various instruments")

DRIFT_BASE = ['BTC','ETH','SOL']

@st.cache_data(ttl=3600)
def get_snowflake_data(query, exchange=None):
    # Determine which database to use based on exchange
    if exchange == "drift":
        database = st.secrets["drift"]["database"]
    else:
        database = st.secrets["gerrit"]["database"]
    
    conn = connector.connect(
        user=st.secrets["gerrit"]["user"],
        password=st.secrets["gerrit"]["password"],
        account=st.secrets["gerrit"]["account"],
        warehouse=st.secrets["gerrit"]["warehouse"],
        database=database,
        role=st.secrets["gerrit"]["role"],
        schema=st.secrets["gerrit"]["schema"]
    )
    
    try:
        cur = conn.cursor()
        cur.execute(query)
        data = cur.fetchall()
        columns = [desc[0] for desc in cur.description]
        df = pd.DataFrame(data, columns=columns)
        return df
    finally:
        conn.close()

def get_funding_table_for_exchange(exchange):
    """Return the appropriate funding data table name for the given exchange"""
    return "DRIFT_FUNDING" if exchange == "drift" else "RAW_FUNDING"

@st.cache_data(ttl=3600)
def get_all_exchanges_with_margin(base_asset):
    """Get all exchanges with their margin assets for a given base asset"""
    query = f"""
    SELECT DISTINCT exchange, margin_asset, symbol, base_token FROM perp_markets 
    WHERE type = 'perpetual' AND base_token = '{base_asset}' 
    ORDER BY exchange, margin_asset
    """
    df = get_snowflake_data(query)
    if base_asset in DRIFT_BASE:
        df = pd.concat([df, pd.DataFrame({'EXCHANGE': ['drift'], 'MARGIN_ASSET': ['USDC'], 
                                          'SYMBOL': [f'{base_asset}-PERP'], 'BASE_TOKEN': [base_asset]})])
    return df

def calculate_funding_averages(df_a, df_b, exchange_a, exchange_b):
    """Calculate average funding rates for different time periods with proper compounding and APY"""
    if df_a.empty or df_b.empty:
        return None
    
    # Convert timestamps to datetime if they aren't already
    if not pd.api.types.is_datetime64_any_dtype(df_a['TIMESTAMP_UTC']):
        df_a['TIMESTAMP_UTC'] = pd.to_datetime(df_a['TIMESTAMP_UTC'])
    if not pd.api.types.is_datetime64_any_dtype(df_b['TIMESTAMP_UTC']):
        df_b['TIMESTAMP_UTC'] = pd.to_datetime(df_b['TIMESTAMP_UTC'])
    
    # Define funding periods per day for different exchanges
    funding_periods = {
        "drift": 24,
        "binance": 3,
        "bybit": 3,
        "hyperliquid": 24,
        "dydx": 24 
    }
    
    # Extract exchange name without margin asset
    exchange_a_name = exchange_a.split(" (")[0].lower()
    exchange_b_name = exchange_b.split(" (")[0].lower()
    
    # Use default if exchange not found
    periods_per_day_a = funding_periods.get(exchange_a_name, 3)
    periods_per_day_b = funding_periods.get(exchange_b_name, 3)
    
    # Calculate mean funding rates
    mean_rate_a = df_a['FUNDING_RATE'].mean()
    mean_rate_b = df_b['FUNDING_RATE'].mean()
    
    # Calculate APY directly using the compound interest formula
    apy_a = (1 + mean_rate_a) ** (periods_per_day_a * 365) - 1
    apy_b = (1 + mean_rate_b) ** (periods_per_day_b * 365) - 1
    
    # Calculate rates for different periods using proper compounding
    # For Exchange A (long position)
    hourly_a = mean_rate_a * periods_per_day_a / 24  # Convert to hourly rate
    daily_a = (1 + hourly_a) ** 24 - 1              # Compound hourly to daily
    weekly_a = (1 + daily_a) ** 7 - 1               # Compound daily to weekly
    monthly_a = (1 + daily_a) ** 30 - 1             # Compound daily to monthly (approx)
    yearly_a = (1 + daily_a) ** 365 - 1             # Compound daily to yearly
    
    # For Exchange B (short position)
    hourly_b = mean_rate_b * periods_per_day_b / 24  # Convert to hourly rate
    daily_b = (1 + hourly_b) ** 24 - 1              # Compound hourly to daily
    weekly_b = (1 + daily_b) ** 7 - 1               # Compound daily to weekly
    monthly_b = (1 + daily_b) ** 30 - 1             # Compound daily to monthly (approx)
    yearly_b = (1 + daily_b) ** 365 - 1             # Compound daily to yearly
    
    # Calculate spreads (long - short)
    # For a funding rate arbitrage, we want to know the spread between
    # paying funding on one exchange (short) and receiving on another (long)
    # Note: For shorts, we negate the rate since we're paying it
    hourly_spread = hourly_b - hourly_a
    daily_spread = daily_b - daily_a
    weekly_spread = weekly_b - weekly_a
    monthly_spread = monthly_b - monthly_a
    yearly_spread = yearly_b - yearly_a
    
    # Calculate the net APY for the spread (compounded)
    apy_spread = apy_b - apy_a
    
    # Create a DataFrame for the comparison table
    comparison_df = pd.DataFrame({
        'Time Period': ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly', 'APY (Compounded)'],
        f'{exchange_a} (Long)': [hourly_a, daily_a, weekly_a, monthly_a, yearly_a, apy_a],
        f'{exchange_b} (Short)': [hourly_b, daily_b, weekly_b, monthly_b, yearly_b, apy_b],
        'Spread (Net)': [hourly_spread, daily_spread, weekly_spread, monthly_spread, yearly_spread, apy_spread]
    })
    
    return comparison_df

st.sidebar.header("Filters")

@st.cache_data(ttl=3600)
def get_base_assets():
    """
        Gerrit instruction: get all base assets that exist on hyperliquid and dydx
    """
    query = """
    SELECT base_token FROM perp_markets WHERE exchange = 'hyperliquid'
    INTERSECT
    SELECT base_token FROM perp_markets WHERE exchange = 'dydx'
    ORDER BY base_token
    """
    return get_snowflake_data(query)['BASE_TOKEN'].unique().tolist()

# Load exchanges and instruments
try:
    base_assets = ['Select an asset'] + get_base_assets()
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)
    selected_base_asset = st.sidebar.selectbox("Instrument", base_assets)

    if selected_base_asset != 'Select an asset':
        # Get all exchanges with their margin assets
        exchanges_df = get_all_exchanges_with_margin(selected_base_asset)
        
        # Create a formatted list of exchanges with margin assets in parentheses
        exchanges_with_margin = []
        exchange_to_margin = {}  # Dictionary to store exchange-to-margin mapping
        
        for _, row in exchanges_df.iterrows():
            exchange_name = row['EXCHANGE']
            margin_asset = row['MARGIN_ASSET']
            display_name = f"{exchange_name} ({margin_asset})"
            exchanges_with_margin.append(display_name)
            exchange_to_margin[display_name] = {
                'exchange': exchange_name,
                'margin_asset': margin_asset,
                'symbol': row['SYMBOL']
            }
        
        # First select the long exchange
        exchange_a = st.sidebar.selectbox("Long Exchange", exchanges_with_margin, index=0)
        
        # Filter out the selected long exchange for the short exchange selection
        short_exchanges = [ex for ex in exchanges_with_margin if ex != exchange_a]
        
        # Select exchange B, default to first available
        default_index_b = 0
        exchange_b = st.sidebar.selectbox("Short Exchange", short_exchanges, index=default_index_b)
    
    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        if selected_base_asset == 'Select an asset':
            st.error("Please select an asset")
            st.stop()
        if not exchange_a or not exchange_b:
            st.error("Please select both long and short exchanges")
            st.stop()

        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
        
        # Extract exchange names and margin assets
        exchange_a_name = exchange_to_margin[exchange_a]['exchange']
        exchange_b_name = exchange_to_margin[exchange_b]['exchange']
        margin_asset_a = exchange_to_margin[exchange_a]['margin_asset']
        margin_asset_b = exchange_to_margin[exchange_b]['margin_asset']
        
        table_a = get_funding_table_for_exchange(exchange_a_name)
        table_b = get_funding_table_for_exchange(exchange_b_name)

        # Get symbols for the exchanges
        symbol_a = exchange_to_margin[exchange_a]['symbol']
        symbol_b = exchange_to_margin[exchange_b]['symbol']

        # Query for Exchange A (long exchange)
        query_a = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM {table_a}
        WHERE exchange = '{exchange_a_name}'
        AND symbol = '{symbol_a}'
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """
        
        # Query for Exchange B
        query_b = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM {table_b}
        WHERE exchange = '{exchange_b_name}'
        AND symbol = '{symbol_b}'
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """

        with st.spinner("Fetching data..."):
            df_a = get_snowflake_data(query_a, exchange_a_name)
            df_b = get_snowflake_data(query_b, exchange_b_name)
        
        # Display data info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{exchange_a} Funding Rates")
            st.write(f"Records: {len(df_a)}")
            if not df_a.empty:
                st.dataframe(df_a.head())
        
        with col2:
            st.subheader(f"{exchange_b} Funding Rates")
            st.write(f"Records: {len(df_b)}")
            if not df_b.empty:
                st.dataframe(df_b.head())
        
        # Plot comparisons
        st.subheader(f"Funding Rate Comparison for {selected_base_asset}")
        
        if not df_a.empty and not df_b.empty:
            # Create figure with secondary y-axis
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            # Add traces
            fig.add_trace(
                go.Scatter(
                    x=df_a['TIMESTAMP_UTC'],
                    y=df_a['FUNDING_RATE'],
                    name=f"{exchange_a}",
                    line=dict(color="blue")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_b['TIMESTAMP_UTC'], 
                    y=df_b['FUNDING_RATE'],
                    name=f"{exchange_b}",
                    line=dict(color="red")
                )
            )
            
            # Set x-axis title
            fig.update_xaxes(title_text="Date")
            
            # Set y-axes titles
            fig.update_yaxes(title_text="Funding Rate (%)",
                             tickformat=".6f")
            
            fig.update_layout(
                title=f"{selected_base_asset} Funding Rate: {exchange_a} vs {exchange_b}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(df_a) > 1 and len(df_b) > 1:
                st.subheader("Statistical Analysis")
                
                df_a_renamed = df_a.rename(columns={"FUNDING_RATE": f"{exchange_a}_rate"})
                df_b_renamed = df_b.rename(columns={"FUNDING_RATE": f"{exchange_b}_rate"})
                
                # Create merged_df with inner join for visualization
                merged_df_inner = pd.merge(
                    df_a_renamed, 
                    df_b_renamed, 
                    on="TIMESTAMP_UTC", 
                    how="inner",
                    suffixes=('_a', '_b')
                )
                
                # Create complete merged_df with outer join for download
                merged_df_complete = pd.merge(
                    df_a_renamed, 
                    df_b_renamed, 
                    on="TIMESTAMP_UTC", 
                    how="outer",
                    suffixes=('_a', '_b')
                )
                merged_df_complete.sort_values("TIMESTAMP_UTC", inplace=True)
                
                if not merged_df_inner.empty:
                    correlation = merged_df_inner[f"{exchange_a}_rate"].corr(merged_df_inner[f"{exchange_b}_rate"])
                    st.write(f"Correlation between {exchange_a} and {exchange_b} funding rates for {selected_base_asset}: {correlation:.4f}")
                    
                    # Scatter plot
                    fig_scatter = px.scatter(
                        merged_df_inner, 
                        x=f"{exchange_a}_rate", 
                        y=f"{exchange_b}_rate",
                        title=f"{selected_base_asset} Funding Rate Correlation: {exchange_a} vs {exchange_b}",
                        labels={
                            f"{exchange_a}_rate": f"{exchange_a} Funding Rate (%)",
                            f"{exchange_b}_rate": f"{exchange_b} Funding Rate (%)"
                        },
                        trendline="ols"
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    # Download data - using the complete merged dataset with outer join
                    st.download_button(
                        label="Download Comparison Data",
                        data=merged_df_complete.to_csv(index=False),
                        file_name=f"{selected_base_asset}_{exchange_a}_vs_{exchange_b}_comparison.csv",
                        mime="text/csv"
                    )

            # Add funding rate comparison table
            st.subheader(f"Funding Rate Comparison Table for {selected_base_asset}")
            
            # Get funding averages
            avg_rates_df = calculate_funding_averages(df_a, df_b, exchange_a, exchange_b)
            
            if avg_rates_df is not None:
                # Format percentages
                for col in avg_rates_df.columns[1:]:  # Skip the 'Time Period' column
                    avg_rates_df[col] = avg_rates_df[col].map('{:.6%}'.format)
                
                # Display the table
                st.dataframe(avg_rates_df)
                
                # Add download button for the table
                st.download_button(
                    label="Download Rate Comparison Table",
                    data=avg_rates_df.to_csv(index=False),
                    file_name=f"{selected_base_asset}_{exchange_a}_vs_{exchange_b}_rate_table.csv",
                    mime="text/csv"
                )
        elif df_a.empty and df_b.empty:
            st.warning(f"No data found for {selected_base_asset} on either exchange for the selected date range.")
        elif df_a.empty:
            st.warning(f"No data found for {selected_base_asset} on {exchange_a} for the selected date range.")
        else:
            st.warning(f"No data found for {selected_base_asset} on {exchange_b} for the selected date range.")

except Exception as e:
    st.error(f"Error connecting to Snowflake: {e}")
    st.info("Please check your Snowflake credentials and connection details.")

# Footer
st.markdown("---")
st.markdown("Funding Rate Comparison Tool | Data from Snowflake")