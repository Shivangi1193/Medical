import pandas as pd

def preprocess_data(df):
    # Handle missing values
    df["specialty"] = df["specialty"].fillna("Other")
    df["disability"] = df["disability"].fillna("None")
    df["place"] = df["place"].fillna("Unknown")
    df["age"] = df["age"].fillna(df["age"].median())

    weather_cols = ["average_temp_day","average_rain_day","max_temp_day","max_rain_day"]
    for col in weather_cols:
        df[col] = df[col].fillna(df[col].median())

    # Drop duplicates
    #df = df.drop_duplicates()

  # Clean mapping using .replace() so it won't break if run twice
    df['gender'] = df['gender'].replace({'F': 0, 'M': 1, 'I': 0})
    df['no_show'] = df['no_show'].replace({'no': 0, 'yes': 1})

    # Force the data types to integers
    df['gender'] = df['gender'].astype(int)
    df['no_show'] = df['no_show'].astype(int)

    # Feature engineering
    df["chronic_conditions"] = df[["Hipertension","Diabetes","Alcoholism","Handcap"]].sum(axis=1)
    df["appointment_time"] = pd.to_datetime(df["appointment_time"])
    df["appointment_date_continuous"] = pd.to_datetime(df["appointment_date_continuous"])
    df["DaysWaiting"] = (df["appointment_date_continuous"] - df["appointment_time"]).dt.days
    df["Weekday"] = df["appointment_date_continuous"].dt.day_name()

    return df
