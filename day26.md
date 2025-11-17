# Day 26: Time Series Forecasting Practice


## Topics
- Sales forecasting
- Seasonality detection
- Model tuning
- Evaluation

## Journal

Forecasted retail sales using ARIMA and Prophet. Compared approaches:
- ARIMA: Better for short-term forecasts
- Prophet: Handles holidays and seasonality well

Key metrics: MAE, RMSE, MAPE

```python
from fbprophet import Prophet

# Prophet model
model = Prophet()
model.fit(df)
future = model.make_future_dataframe(periods=12, freq='M')
forecast = model.predict(future)
model.plot(forecast)
```

## Reflections
Time series forecasting requires understanding business cycles. External factors (holidays, promotions) significantly impact accuracy. Ensemble approaches often perform best.

## Resources
- [Prophet Documentation](https://facebook.github.io/prophet/)
