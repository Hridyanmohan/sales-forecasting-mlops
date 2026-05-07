import pandas as pd

def process_data():
    df = pd.read_excel("data/Forecasting Case- Study.xlsx")
    print(df['Total'].describe())
    df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
    df = df.sort_values(['State', 'Date'])

    #Lag features (t-1, t-7, t-30)
    for lag in [1, 7, 30]:
        df[f'lag_{lag}'] = df.groupby('State')['Total'].shift(lag)

    # Rolling mean / std
    df['roll_mean_7'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(7).mean())
    df['roll_std_7'] = df.groupby('State')['Total'].transform(lambda x: x.rolling(7).std())

    #Temporal features
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek

    df.dropna().to_csv("data/processed_data.csv", index=False)
    print("Preprocessing Complete!")

if __name__ == "__main__":
    process_data()