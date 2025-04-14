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
def get_exchanges(base_asset, margin_asset):
    query = f"""
    SELECT DISTINCT exchange,symbol,base_token,margin_asset FROM perp_markets 
    WHERE margin_asset = '{margin_asset}' AND type = 'perpetual' 
    AND base_token = '{base_asset}' ORDER BY exchange
    """
    df = get_snowflake_data(query)
    if base_asset in DRIFT_BASE:
        df = pd.concat([df, pd.DataFrame({'EXCHANGE': ['drift'], 'SYMBOL': [f'{base_asset}-PERP'], 'BASE_TOKEN': [base_asset], 'MARGIN_ASSET': ['USDT']})])
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
    
    # Use default if exchange not found
    periods_per_day_a = funding_periods.get(exchange_a.lower(), 3)
    periods_per_day_b = funding_periods.get(exchange_b.lower(), 3)
    
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
    hourly_spread = hourly_a - (-hourly_b)
    daily_spread = daily_a - (-daily_b)
    weekly_spread = weekly_a - (-weekly_b)
    monthly_spread = monthly_a - (-monthly_b)
    yearly_spread = yearly_a - (-yearly_b)
    
    # Calculate the net APY for the spread (compounded)
    apy_spread = apy_a - (-apy_b)
    
    # Create a DataFrame for the comparison table
    comparison_df = pd.DataFrame({
        'Time Period': ['Hourly', 'Daily', 'Weekly', 'Monthly', 'Yearly', 'APY (Compounded)'],
        f'{exchange_a} (Long)': [hourly_a, daily_a, weekly_a, monthly_a, yearly_a, apy_a],
        f'{exchange_b} (Short)': [-hourly_b, -daily_b, -weekly_b, -monthly_b, -yearly_b, -apy_b],
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
        exchanges_df = get_exchanges(selected_base_asset, 'USDT')  # Initial fetch with default margin asset
        exchanges_list = exchanges_df['EXCHANGE'].unique().tolist()
        
        # First select the long exchange
        exchange_a = st.sidebar.selectbox("Long Exchange", exchanges_list, index=0)
        
        # Then select the margin asset for exchange A (long)
        margin_asset_a = st.sidebar.selectbox("Long Exchange Margin Asset", ['USDT', 'USDC'], key="margin_a")
        
        # Update the exchanges list based on the selected base asset and margin asset for exchange A
        exchanges_df_a = get_exchanges(selected_base_asset, margin_asset_a)
        exchanges_list_a = exchanges_df_a['EXCHANGE'].unique().tolist()
        
        # If exchange_a is not in the updated list, reset it
        if exchange_a not in exchanges_list_a and exchanges_list_a:
            exchange_a = exchanges_list_a[0]
        
        # For exchange B, default margin asset to match exchange A
        margin_asset_b = st.sidebar.selectbox("Short Exchange Margin Asset", 
                                            ['USDT', 'USDC'], 
                                            index=['USDT', 'USDC'].index(margin_asset_a),
                                            key="margin_b")
        
        # Update exchanges list for exchange B based on its margin asset
        exchanges_df_b = get_exchanges(selected_base_asset, margin_asset_b)
        exchanges_list_b = exchanges_df_b['EXCHANGE'].unique().tolist()
        
        # Select exchange B, default to index 1 if available, otherwise 0
        default_index_b = min(1, len(exchanges_list_b)-1) if exchanges_list_b else 0
        exchange_b = st.sidebar.selectbox("Short Exchange", exchanges_list_b, index=default_index_b)
    
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
        table_a = get_funding_table_for_exchange(exchange_a)
        table_b = get_funding_table_for_exchange(exchange_b)

        # Use the specific margin assets for each exchange
        a_df = exchanges_df_a[(exchanges_df_a['EXCHANGE'] == exchange_a) & 
                            (exchanges_df_a['BASE_TOKEN'] == selected_base_asset) & 
                            (exchanges_df_a['MARGIN_ASSET'] == margin_asset_a)]
        
        b_df = exchanges_df_b[(exchanges_df_b['EXCHANGE'] == exchange_b) & 
                            (exchanges_df_b['BASE_TOKEN'] == selected_base_asset) & 
                            (exchanges_df_b['MARGIN_ASSET'] == margin_asset_b)]

        if len(a_df) == 0:
            st.error(f"No data found for {selected_base_asset} with margin asset {margin_asset_a} on {exchange_a} for the selected date range.")
            st.stop()
        if len(b_df) == 0:
            st.error(f"No data found for {selected_base_asset} with margin asset {margin_asset_b} on {exchange_b} for the selected date range.")
            st.stop()

        market_symbols_a = "', '".join(a_df['SYMBOL'].tolist())
        market_symbols_b = "', '".join(b_df['SYMBOL'].tolist())

        # Query for Exchange A (long exchange)
        query_a = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM {table_a}
        WHERE exchange = '{exchange_a}'
        AND symbol IN ('{market_symbols_a}')
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """
        
        # Query for Exchange B
        query_b = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM {table_b}
        WHERE exchange = '{exchange_b}'
        AND symbol IN ('{market_symbols_b}')
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """

        with st.spinner("Fetching data..."):
            df_a = get_snowflake_data(query_a, exchange_a)
            df_b = get_snowflake_data(query_b, exchange_b)
        
        # Get the first symbol for each exchange for display purposes
        symbol_a = df_a['SYMBOL'].iloc[0] if not df_a.empty and len(df_a) > 0 else f"{selected_base_asset}-PERP"
        symbol_b = df_b['SYMBOL'].iloc[0] if not df_b.empty and len(df_b) > 0 else f"{selected_base_asset}-PERP"
        
        # Display data info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{exchange_a} Funding Rates for {symbol_a}")
            st.write(f"Records: {len(df_a)}")
            if not df_a.empty:
                st.dataframe(df_a.head())
        
        with col2:
            st.subheader(f"{exchange_b} Funding Rates for {symbol_b}")
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
                    name=f"{exchange_a} ({symbol_a})",
                    line=dict(color="blue")
                )
            )
            
            fig.add_trace(
                go.Scatter(
                    x=df_b['TIMESTAMP_UTC'], 
                    y=df_b['FUNDING_RATE'],
                    name=f"{exchange_b} ({symbol_b})",
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
                            f"{exchange_a}_rate": f"{exchange_a} Funding Rate (%) - {symbol_a}",
                            f"{exchange_b}_rate": f"{exchange_b} Funding Rate (%) - {symbol_b}"
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