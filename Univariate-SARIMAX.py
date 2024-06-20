import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import statsmodels.api as sm

def get_data() -> pd.DataFrame:
    """
    Reads monthly car sales data from a local CSV file or downloads it from a URL.

    Returns:
        pd.DataFrame: The monthly car sales data.
    """
    data_path = Path("./monthly-car-sales.csv")
    if data_path.is_file():
        df: pd.DataFrame = pd.read_csv(
            data_path, header=0, index_col=0, parse_dates=[0]
        )
    else:
        data_url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-car-sales.csv"
        df: pd.DataFrame = pd.read_csv(data_url, header=0, index_col=0, parse_dates=[0])
        df.to_csv(data_path, index=False)
    return df


if __name__ == "__main__":
    # first check if the file exists

    df = get_data()
    df.index = df.index.to_period("M")

    # split into train and test
    train_size = int(df.shape[0] * 0.8)
    values = df.values
    train, test = values[:train_size], values[train_size:]
    history = [x for x in train]
    predictions = list()
    
    for t in range(len(test)):
        model = sm.tsa.statespace.SARIMAX(history, order=(5, 1, 0), seasonal_order=(0, 1, 1, 12))
        model_fit = model.fit()
        output = model_fit.forecast()
        yhat = output[0]
        predictions.append(yhat)
        obs = test[t]
        history.append(obs)
        print(f"predicted={yhat}, expected={obs[0]}")
    
    # evaluate forecasts
    rmse = np.sqrt(np.mean((np.array(predictions) - np.array(test))**2))
    print(f"Test RMSE: {rmse}")
    
    # plot forecasts against actual outcomes
    plt.plot(test)
    plt.plot(predictions, color="red")
    plt.show()