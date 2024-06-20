import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def get_data() -> pd.DataFrame:
    """
    Reads monthly car sales data from a local CSV file or downloads it from a URL.

    Returns:
        pd.DataFrame: The monthly car sales data.
    """
    data_path = Path("./monthly-car-sales.csv")
    if data_path.is_file():
        df: pd.DataFrame = pd.read_csv(data_path, header=0, index_col=0, parse_dates=[0])
    else:
        data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
        df: pd.DataFrame = pd.read_csv(data_url, header=0, index_col=0, parse_dates=[0])
        df.to_csv(data_path, index=False)
    return df


if __name__ == "__main__":
    # first check if the file exists

    df = get_data()
    df.plot()
    plt.show()
    
    # this data has a clear trend and not stationary yet 
    # we need to make it stationary lets plot the autocorrelation
    
    pd.plotting.autocorrelation_plot(df)
    plt.show()
    
