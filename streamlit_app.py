import streamlit as st
import pandas as pd
import plotly.express as px
import snowflake.connector as connector
from datetime import datetime, timedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Funding Rate Comparison", layout="wide")

# App title and introduction
st.title("Crypto Funding Rate Comparison")
st.write("Compare funding rates between exchanges for BTC, ETH, and SOL instruments")

# Function to connect to Snowflake and get data
@st.cache_data(ttl=3600)  # Cache the function with a time-to-live of 1 hour
def get_snowflake_data(query):
    # Replace with your Snowflake credentials
    conn = connector.connect(
        user=st.secrets["snowflake"]["user"],
        password=st.secrets["snowflake"]["password"],
        account=st.secrets["snowflake"]["account"],
        warehouse=st.secrets["snowflake"]["warehouse"],
        database=st.secrets["snowflake"]["database"],
        role=st.secrets["snowflake"]["role"],
        schema=st.secrets["snowflake"]["schema"]
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

# Function to get available exchanges
@st.cache_data(ttl=3600)
def get_exchanges():
    query = "SELECT DISTINCT exchange FROM raw_funding ORDER BY exchange"
    return get_snowflake_data(query)

# Function to get available instruments
@st.cache_data(ttl=3600)
def get_instruments():
    query = """
        SELECT DISTINCT symbol 
        FROM raw_funding 
        WHERE symbol LIKE 'SOL%' 
           OR symbol LIKE 'ETH%'
           OR symbol LIKE 'BTC%'
        ORDER BY symbol
    """
    return get_snowflake_data(query)

# Sidebar for filters
st.sidebar.header("Filters")

# Load exchanges and instruments
try:
    exchanges_df = get_exchanges()
    instruments_df = get_instruments()
    
    # Convert to lists for dropdown
    exchanges_list = exchanges_df['EXCHANGE'].tolist()
    
    # Simplify to just base assets (ETH, BTC, SOL)
    base_assets = ["BTC", "ETH", "SOL"]
    
    # Date range selection
    default_end_date = datetime.now()
    default_start_date = default_end_date - timedelta(days=30)
    
    start_date = st.sidebar.date_input("Start Date", default_start_date)
    end_date = st.sidebar.date_input("End Date", default_end_date)
    
    # Exchange selection
    exchange_a = st.sidebar.selectbox("Exchange A", exchanges_list, index=0)
    exchange_b = st.sidebar.selectbox("Exchange B", exchanges_list, index=1 if len(exchanges_list) > 1 else 0)
    
    # Base asset selection (BTC, ETH, SOL)
    selected_base_asset = st.sidebar.selectbox("Instrument", base_assets)
    
    # Fetch data button
    if st.sidebar.button("Fetch Data"):
        # Convert dates to strings for query
        start_date_str = start_date.strftime("%Y-%m-%dT00:00:00Z")
        end_date_str = end_date.strftime("%Y-%m-%dT23:59:59Z")
        
        # Query for Exchange A
        query_a = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM raw_funding
        WHERE exchange = '{exchange_a}'
        AND symbol LIKE '{selected_base_asset}%'
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """
        
        # Query for Exchange B
        query_b = f"""
        SELECT timestamp_utc, funding_rate, symbol
        FROM raw_funding
        WHERE exchange = '{exchange_b}'
        AND symbol LIKE '{selected_base_asset}%'
        AND timestamp_utc BETWEEN '{start_date_str}' AND '{end_date_str}'
        ORDER BY timestamp_utc
        """
        
        # Get data
        with st.spinner("Fetching data..."):
            df_a = get_snowflake_data(query_a)
            df_b = get_snowflake_data(query_b)
        
        # Get the first symbol for each exchange for display purposes
        symbol_a = df_a['SYMBOL'].iloc[0] if not df_a.empty and len(df_a) > 0 else f"{selected_base_asset}-perp"
        symbol_b = df_b['SYMBOL'].iloc[0] if not df_b.empty and len(df_b) > 0 else f"{selected_base_asset}-perp"
        
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
                    # Calculate correlation
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
                    
                    # Statistical summary
                    st.subheader("Summary Statistics")
                    summary_df = pd.DataFrame({
                        "Metric": ["Mean", "Median", "Min", "Max", "Std Dev"],
                        f"{exchange_a} ({symbol_a})": [
                            f"{df_a['FUNDING_RATE'].mean():.6f}",
                            f"{df_a['FUNDING_RATE'].median():.6f}",
                            f"{df_a['FUNDING_RATE'].min():.6f}",
                            f"{df_a['FUNDING_RATE'].max():.6f}",
                            f"{df_a['FUNDING_RATE'].std():.6f}"
                        ],
                        f"{exchange_b} ({symbol_b})": [
                            f"{df_b['FUNDING_RATE'].mean():.6f}",
                            f"{df_b['FUNDING_RATE'].median():.6f}",
                            f"{df_b['FUNDING_RATE'].min():.6f}",
                            f"{df_b['FUNDING_RATE'].max():.6f}",
                            f"{df_b['FUNDING_RATE'].std():.6f}"
                        ]
                    })
                    
                    st.table(summary_df)
                    
                    # Download data
                    st.download_button(
                        label="Download Comparison Data",
                        data=merged_df.to_csv(index=False),
                        file_name=f"{selected_base_asset}_{exchange_a}_vs_{exchange_b}_comparison.csv",
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