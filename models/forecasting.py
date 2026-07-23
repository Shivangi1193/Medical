from prophet import Prophet
from sklearn.metrics import mean_absolute_percentage_error, r2_score

def train_forecast(df):
    daily_counts = df.groupby("appointment_date_continuous").size().reset_index(name="y")
    daily_counts.rename(columns={"appointment_date_continuous":"ds"}, inplace=True)

    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(daily_counts)

    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    y_true = daily_counts["y"]
    y_pred = forecast.loc[:len(y_true)-1,"yhat"]

    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return model, forecast, mape, r2
