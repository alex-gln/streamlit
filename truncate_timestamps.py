import pandas as pd

def truncate_timestamps(file_path):
    # Read the CSV file
    df = pd.read_csv(file_path)
    
    # Check if timestamp_utc column exists
    if 'TIMESTAMP_UTC' in df.columns:
        # Convert to datetime and truncate to hour
        df['TIMESTAMP_UTC'] = pd.to_datetime(df['TIMESTAMP_UTC']).dt.strftime('%Y-%m-%d %H:00:00')
    
    # Save back to the same file
    df.to_csv(file_path, index=False)
    
    print(f"Timestamps in {file_path} truncated to hour level.")

# Example usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        truncate_timestamps(file_path)
    else:
        print("Please provide a CSV file path as an argument.")
        print("Example: python truncate_timestamps.py data.csv")
