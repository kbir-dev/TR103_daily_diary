# Day 21: Time Series Forecasting: ARIMA

## Topics
- Time series components
- ARIMA model
- Model selection
- Forecasting

## Journal

Explored ARIMA (AutoRegressive Integrated Moving Average) for time series forecasting. Key parameters: p (AR), d (differencing), q (MA). Used ACF/PACF plots for parameter selection.

```python
from statsmodels.tsa.arima.model import ARIMA
import matplotlib.pyplot as plt

# Sample data
air_passengers = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv')

# Fit ARIMA model
model = ARIMA(air_passengers['Passengers'], order=(5,1,0))
model_fit = model.fit()

# Forecast
forecast = model_fit.forecast(steps=12)
plt.plot(air_passengers['Passengers'])
plt.plot(forecast, color='red')
```

## Reflections
ARIMA models capture temporal dependencies but require stationary data. Differencing handles trends. Model selection involves balancing fit and complexity.

## Resources
- [Statsmodels ARIMA](https://www.statsmodels.org/stable/tsa.html)
