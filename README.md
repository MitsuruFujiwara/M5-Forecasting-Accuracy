# M5-Forecasting-Accuracy
This repository is my solition for kaggle M5 Forecasting Accuracy. See [competition website](https://www.kaggle.com/c/m5-forecasting-accuracy) for the details.

### Result
- Local cv score: 0.56874
- Private LB score: 0.60101

### Features
- Basic statistics features: 7~28days rolling statistics.
- Lag features: 0~28days lag.
- Price features: rolling stat, momentum etc.
- Calendar features: seasonality, event flag, holiday flag etc.
- Other features: days from release, zero sales ratio (7~28days).

### Cross Validation
- Custom timeseries split (3fold): last 2 month (fold1,2) + 1 year before (fold3)
- Group k-fold (5fold): group = year + month

### Models
LightGBM with tweedie loss. Blending 4 models with different cv strategies (TS or group) and different data split (by category or by week or no split).

|name|cv|split|fold1|fold2|fold3|avg|weight|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Weekly Group|Group|Week|0.60118|0.50369|0.65736|0.58741|0.4175|
|Weekly TS|TS|Week|0.61178|0.53015|0.68283|0.60825|0.2320|
|Category Group|Group|Category|0.62392|0.48934|0.69283|0.60203|0.2052|
|No split Group|Group|-|0.60705|0.49313|0.66869|0.58963|0.1732|
|Blend|-|-|0.54787|0.51075|0.64282|0.56715|-|
