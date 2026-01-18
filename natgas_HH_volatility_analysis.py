import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def read_csv_file(file_path: str) -> pd.DataFrame:
    """
    Read a CSV file and return as a pandas DataFrame.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: The data from the CSV file
    """
    try:
        df = pd.read_csv(file_path, delimiter=',', engine='python')
        return df
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None

# Reverse the dataframe order to get correct chronological date ordering
def convert_date_format(df: pd.DataFrame, date_column: str) -> pd.DataFrame:
    """
    Convert date format from US (MM/DD/YYYY) to UK (DD/MM/YYYY).
    
    Args:
        df (pd.DataFrame): The dataframe containing the date column
        date_column (str): Name of the date column
        
    Returns:
        pd.DataFrame: DataFrame with converted date format
    """
    df_copy = df.copy()
    df_copy[date_column] = pd.to_datetime(df_copy[date_column], format='%m/%d/%Y').dt.strftime('%d/%m/%Y')
    return df_copy

def plot_price_over_time(df: pd.DataFrame, date_column: str, price_column: str) -> None:
    """
    Plot price data over time.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        date_column (str): Name of the date column
        price_column (str): Name of the price column
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert date column to datetime for proper spacing
    dates = pd.to_datetime(df[date_column], format='%d/%m/%Y')
    
    ax.plot(dates, df[price_column], linewidth=2.5, color='#1f77b4', label='Spot Price')
    ax.fill_between(dates, df[price_column], alpha=0.15, color='#1f77b4')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Price ($/MMBtu)', fontsize=12, fontweight='bold')
    ax.set_title('Henry Hub Natural Gas Spot Price Over Time', fontsize=16, fontweight='bold', pad=20)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=12))  # Every 12 months
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.show()

def calculate_ln_returns(df: pd.DataFrame, price_column: str) -> pd.DataFrame:
    """
    Calculate logarithmic returns from price data.
    
    Args:
        df (pd.DataFrame): The dataframe containing the price data
        price_column (str): Name of the price column
        
    Returns:
        pd.DataFrame: DataFrame with an added 'ln_returns' column
    """
    df_copy = df.copy()
    df_copy['ln_returns'] = np.log(df_copy[price_column] / df_copy[price_column].shift(1))
    return df_copy

def calculate_rolling_volatility(df: pd.DataFrame, returns_column: str, window: int) -> pd.DataFrame:
    """
    Calculate rolling standard deviation (volatility) of returns.
    
    Args:
        df (pd.DataFrame): The dataframe containing the returns data
        returns_column (str): Name of the returns column
        window (int): Number of samples to use for the rolling window
        
    Returns:
        pd.DataFrame: DataFrame with an added 'rolling_volatility' column
    """
    df_copy = df.copy()
    df_copy['rolling_volatility'] = df_copy[returns_column].rolling(window=window).std()
    return df_copy

def plot_volatility_over_time(df: pd.DataFrame, date_column: str, volatility_column: str, window: int = None) -> None:
    """
    Plot volatility data over time.
    
    Args:
        df (pd.DataFrame): The dataframe containing the data
        date_column (str): Name of the date column
        volatility_column (str): Name of the volatility column
        window (int): Rolling window size used for volatility calculation
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Convert date column to datetime for proper spacing
    dates = pd.to_datetime(df[date_column], format='%d/%m/%Y')
    
    ax.plot(dates, df[volatility_column], linewidth=2.5, color='#ff8c00', label='Rolling Volatility')
    ax.fill_between(dates, df[volatility_column], alpha=0.2, color='#ff8c00')
    
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Rolling Volatility', fontsize=12, fontweight='bold')
    title = 'Henry Hub Natural Gas Volatility Over Time'
    if window:
        title += f' (Window: {window} weeks)'
    ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
    ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator(interval=12))  # Every 12 months
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y-%m'))
    plt.xticks(rotation=45, fontsize=10)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.show()

def main():
    """Main entry point for the natgas volatility analysis."""
    # Replace with your actual CSV file path
    csv_file_path = "Henry_Hub_Natural_Gas_Spot_Price.csv"
    
    df = read_csv_file(csv_file_path)
    # Reverse the dataframe order to get correct chronological date ordering
    df = df.iloc[::-1].reset_index(drop=True)
    df = convert_date_format(df, 'Week of')
    if df is not None:
        print(f"Successfully loaded data with shape: {df.shape}")
        print(df.head())

    plot_price_over_time(df, 'Week of', 'Henry Hub Natural Gas Spot Price Dollars per Million Btu')
    volatility_window = 156
    
    df_with_returns = calculate_ln_returns(df, 'Henry Hub Natural Gas Spot Price Dollars per Million Btu')
    df_with_volatility = calculate_rolling_volatility(df_with_returns, 'ln_returns', volatility_window) 
    
    plot_volatility_over_time(df_with_volatility, 'Week of', 'rolling_volatility', volatility_window)
    

if __name__ == "__main__":
    main()