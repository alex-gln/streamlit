import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector as connector
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots


st.set_page_config(page_title="Funding Rate Comparison", layout="wide")
st.title("Crypto Funding Rate Comparison")
st.write("Compare funding rates between exchanges for BTC, ETH, and SOL instruments")

@st.cache_data(ttl=3600)
def get_snowflake_data(query, exchange=None):
    # Determine which database to use based on exchange
    if exchange == "drift":
        database = st.secrets["drift"]["database"]
        schema = st.secrets["drift"]["schema"]
    else:
        database = st.secrets["gerrit"]["database"]
        schema = st.secrets["gerrit"]["schema"]
    
    conn = connector.connect(
        user=st.secrets["gerrit"]["user"],
        password=st.secrets["gerrit"]["password"],
        account=st.secrets["gerrit"]["account"],
        warehouse=st.secrets["gerrit"]["warehouse"],
        database=database,
        role=st.secrets["gerrit"]["role"],
        schema=schema
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

@st.cache_data(ttl=3600)
def get_exchanges():
    query = "SELECT DISTINCT exchange FROM raw_funding ORDER BY exchange"
    return get_snowflake_data(query)

def get_funding_table_for_exchange(exchange):
    """Return the appropriate funding data table name for the given exchange"""
    return "DRIFT_FUNDING" if exchange == "drift" else "RAW_FUNDING"

@st.cache_data(ttl=3600)
def get_market_symbols(exchange=None, base_asset=None, margin_asset='USDT'):
    # Special case for drift
    if exchange == "drift":
        return pd.DataFrame({'SYMBOL': [f'{base_asset}-PERP']})

    where_clauses = [f"margin_asset = '{margin_asset}'", "type = 'perpetual'"]
    if exchange:
        where_clauses.append(f"exchange = '{exchange}'")
    if base_asset:
        where_clauses.append(f"base_token = '{base_asset}'")
    where_clause = " AND ".join(where_clauses)
    
    query = f"""
    SELECT DISTINCT symbol, margin_asset
    FROM perp_markets
    WHERE {where_clause}
    ORDER BY symbol
    """
    result = get_snowflake_data(query)

    return result

@st.cache_data(ttl=3600)
def get_market_symbols_formatted(exchange, base_asset, margin_asset='USDT'):
    result = get_market_symbols(exchange, base_asset, margin_asset)
    usdt_flag = True
    if result.empty:
        usdt_flag = False
    return "', '".join(result['SYMBOL'].tolist()), result['MARGIN_ASSET'].tolist(), usdt_flag

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
        "drift": 3,      # 8-hour periods (3 per day)
        "binance": 3,    # 8-hour periods (3 per day)
        "bybit": 3,      # 8-hour periods (3 per day)
        "hyperliquid": 24, # 8-hour periods (3 per day)
        "dydx": 24,      # 1-hour periods (24 per day)
        # Add other exchanges as needed
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

# Load exchanges and instruments
try:
    exchanges_df = get_exchanges()
    exchanges_df = pd.concat([exchanges_df, pd.DataFrame({'EXCHANGE': ['drift']})])
    base_assets = ['BTC', 'ETH', 'SOL']
    
    # Convert to lists for dropdown
    exchanges_list = exchanges_df['EXCHANGE'].tolist()
    
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)

    exchange_a = st.sidebar.selectbox("Long Exchange", exchanges_list, index=0)
    exchange_b = st.sidebar.selectbox("Short Exchange", exchanges_list, index=1 if len(exchanges_list) > 1 else 0)
    selected_base_asset = st.sidebar.selectbox("Instrument", base_assets)
    
    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        
        # Get market symbols
        # TODO getting both as USDC if one USDT is not available. 
        # There is an unnecesary query in the worst case
        usdt_flag = True
        if exchange_a == "drift":
            market_symbols_a, margin_assets_a = f"{selected_base_asset}-PERP", ["USDT"]
        else:
            market_symbols_a, margin_assets_a, usdt_long_flag = get_market_symbols_formatted(exchange_a, selected_base_asset, "USDT")
            if not usdt_long_flag:
                market_symbols_a, margin_assets_a, _ = get_market_symbols_formatted(exchange_a, selected_base_asset, "USDC")

        if exchange_b == "drift":
            market_symbols_b, margin_assets_b = f"{selected_base_asset}-PERP", ["USDT"]
        else:
            if usdt_long_flag:
                market_symbols_b, margin_assets_b, usdt_short_flag = get_market_symbols_formatted(exchange_b, selected_base_asset, "USDT")
                if not usdt_short_flag:
                    market_symbols_a, margin_assets_a, _ = get_market_symbols_formatted(exchange_a, selected_base_asset, "USDC")
                    market_symbols_b, margin_assets_b, _ = get_market_symbols_formatted(exchange_b, selected_base_asset, "USDC")
            else:
                market_symbols_b, margin_assets_b, _ = get_market_symbols_formatted(exchange_b, selected_base_asset, "USDC")

        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")

        # Get appropriate table names
        table_a = get_funding_table_for_exchange(exchange_a)
        table_b = get_funding_table_for_exchange(exchange_b)

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
        symbol_a = df_a['SYMBOL'].iloc[0] if not df_a.empty and len(df_a) > 0 else f"{selected_base_asset}-perp"
        margin_asset_a = margin_assets_a[0]
        symbol_b = df_b['SYMBOL'].iloc[0] if not df_b.empty and len(df_b) > 0 else f"{selected_base_asset}-perp"
        margin_asset_b = margin_assets_b[0]
        
        # Display data info
        col1, col2 = st.columns(2)
        with col1:
            st.subheader(f"{exchange_a} Funding Rates for {symbol_a} ({margin_asset_a})")
            st.write(f"Records: {len(df_a)}")
            if not df_a.empty:
                st.dataframe(df_a.head())
        
        with col2:
            st.subheader(f"{exchange_b} Funding Rates for {symbol_b} ({margin_asset_b})")
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
            fig.update_yaxes(title_text="Funding Rate (%)")
            
            fig.update_layout(
                title=f"{selected_base_asset} Funding Rate: {exchange_a} vs {exchange_b}",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=600
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation analysis
            if len(df_a) > 1 and len(df_b) > 1:
                st.subheader("Statistical Analysis")
                
                # Merge dataframes
                df_a_renamed = df_a.rename(columns={"FUNDING_RATE": f"{exchange_a}_rate"})
                df_b_renamed = df_b.rename(columns={"FUNDING_RATE": f"{exchange_b}_rate"})
                
                merged_df = pd.merge(
                    df_a_renamed, 
                    df_b_renamed, 
                    on="TIMESTAMP_UTC", 
                    how="inner",
                    suffixes=('_a', '_b')
                )
                
                if not merged_df.empty:
                    correlation = merged_df[f"{exchange_a}_rate"].corr(merged_df[f"{exchange_b}_rate"])
                    st.write(f"Correlation between {exchange_a} and {exchange_b} funding rates for {selected_base_asset}: {correlation:.4f}")
                    
                    # Scatter plot
                    fig_scatter = px.scatter(
                        merged_df, 
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
                    
                    # Download data
                    st.download_button(
                        label="Download Comparison Data",
                        data=merged_df.to_csv(index=False),
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