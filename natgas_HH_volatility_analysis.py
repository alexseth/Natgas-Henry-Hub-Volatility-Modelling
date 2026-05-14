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
    
def calculate_volatility_by_window(df: pd.DataFrame, returns_column: str, min_window: int, max_window: int, step: int = 1) -> pd.DataFrame:
        """
        Calculate average volatility for multiple rolling window sizes.
        
        Args:
            df (pd.DataFrame): The dataframe containing the returns data
            returns_column (str): Name of the returns column
            min_window (int): Minimum window size
            max_window (int): Maximum window size
            step (int): Step size between window sizes
            
        Returns:
            pd.DataFrame: DataFrame with window sizes and corresponding average volatilities
        """
        results = []
        for window in range(min_window, max_window + 1, step):
            volatility = (df[returns_column].rolling(window=window).std())
            avg_volatility = volatility.mean() # Average variance
            results.append({'window_size': window, 'avg_volatility': avg_volatility})
        return pd.DataFrame(results)

def plot_volatility_by_window(window_df: pd.DataFrame) -> None:
    """
    Plot average volatility across different window sizes.
    
    Args:
        window_df (pd.DataFrame): DataFrame with window_size and avg_volatility columns
    """
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))
    
    ax.plot(window_df['window_size'], window_df['avg_volatility'], linewidth=2.5, color='#2ca02c', marker='o', markersize=4)
    
    ax.set_xlabel('Window Size (weeks)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Average Volatility', fontsize=12, fontweight='bold')
    ax.set_title('Average Volatility vs Rolling Window Size', fontsize=16, fontweight='bold', pad=20)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    fig.tight_layout()
    plt.show()

def calculate_variance_scaling_by_horizon(
    df: pd.DataFrame,
    price_column: str,
    min_horizon: int,
    max_horizon: int,
    step: int = 1,
    use_non_overlapping: bool = True,
    benchmark: str = "newey_west",
    nw_lags: int = 13,
) -> pd.DataFrame:
    """
    Estimate how the variance of horizon-n aggregated log returns scales with horizon length n.

    Empirical series:
        R^(n)_t = ln(P_t / P_{t-n})

    Benchmarks:
    - IID:          Var(R^(n)) ≈ n * Var(1-week return)
    - Newey–West:   Var(R^(n)) ≈ n * LRV   (accounts for autocorrelation)

    Returns:
        DataFrame with empirical variance and correct linear benchmark.
    """
    prices = df[price_column].astype(float)

    if (prices <= 0).any():
        raise ValueError(f"All prices in '{price_column}' must be > 0.")

    logp = np.log(prices)

    # Weekly log returns (demeaned)
    r = logp.diff(1).dropna()
    r = r - r.mean()

    # Lag-0 autocovariance
    gamma0 = np.mean(r.values ** 2)

    # Choose benchmark slope
    if benchmark.lower() == "iid":
        slope = gamma0

    elif benchmark.lower() == "newey_west":
        L = int(nw_lags)
        lrv = gamma0

        for k in range(1, L + 1):
            w = 1.0 - k / (L + 1.0)

            # Sample autocovariance at lag k
            gamma_k = np.mean(
                r.iloc[k:].values * r.iloc[:-k].values
            )

            lrv += 2.0 * w * gamma_k

        slope = lrv

    else:
        raise ValueError("benchmark must be 'iid' or 'newey_west'.")

    results = []
    for n in range(min_horizon, max_horizon + 1, step):
        rn = logp.diff(n).dropna()

        if use_non_overlapping:
            rn = rn.iloc[::n]

        if len(rn) < 10:
            continue

        var_n = np.var(rn.values, ddof=1)
        std_n = np.sqrt(var_n)

        results.append({
            "horizon_weeks": n,
            "var_n_week_log_return": var_n,
            "std_n_week_log_return": std_n,
            "benchmark_var_n": n * slope,
            "benchmark_slope": slope,
            "benchmark_type": benchmark.lower(),
        })

    return pd.DataFrame(results)

def plot_variance_scaling_by_horizon(results_df: pd.DataFrame) -> None:
    """
    Plot empirical variance of n-week log returns against the correct linear benchmark.

    Y-axis: Var( n-week log return )
    X-axis: Horizon n (weeks)

    The dashed line is the *correct* linear scaling benchmark:
      - IID benchmark if benchmark_type == 'iid'
      - Newey–West (autocorrelation-adjusted) if benchmark_type == 'newey_west'
    """
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_smoothing_spline

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    x = results_df["horizon_weeks"].values
    y = results_df["var_n_week_log_return"].values

    ax.scatter(x, y, s=18, alpha=0.4, color="#1f77b4", zorder=2, label="Empirical: Var(n-week log return)")

    spline = make_smoothing_spline(x, y)
    x_smooth = np.linspace(x.min(), x.max(), 300)
    ax.plot(x_smooth, spline(x_smooth), linewidth=2.5, color="#1f77b4", label="Trend (smoothing spline)")

    ax.plot(
        results_df["horizon_weeks"],
        results_df["benchmark_var_n"],
        linestyle="--",
        linewidth=2.2,
        label=f"Linear benchmark ({results_df['benchmark_type'].iloc[0]})"
    )

    ax.set_xlabel("Horizon n (weeks)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Variance", fontsize=12, fontweight="bold")
    ax.set_title(
        "Variance Scaling of Aggregated Returns vs Horizon",
        fontsize=16,
        fontweight="bold",
        pad=20,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(frameon=False)

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

    #plot_price_over_time(df, 'Week of', 'Henry Hub Natural Gas Spot Price Dollars per Million Btu')
    volatility_window = 156
    
    df_with_returns = calculate_ln_returns(df, 'Henry Hub Natural Gas Spot Price Dollars per Million Btu')
    df_with_volatility = calculate_rolling_volatility(df_with_returns, 'ln_returns', volatility_window) 
    #plot_volatility_over_time(df_with_volatility, 'Week of', 'rolling_volatility', window=volatility_window)
    

    window_volatility_df = calculate_volatility_by_window(df_with_returns, 'ln_returns', 4, 500, step=10)
    plot_volatility_by_window(window_volatility_df)
    
    variance_scaling_df = calculate_variance_scaling_by_horizon(
        df,
        price_column='Henry Hub Natural Gas Spot Price Dollars per Million Btu',
        min_horizon=1,
        max_horizon=208,
        step=1,
        use_non_overlapping=True,
        benchmark="newey_west",
        nw_lags=13,
    )
    plot_variance_scaling_by_horizon(variance_scaling_df)
        
    

if __name__ == "__main__":
    main()