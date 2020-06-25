# M5-Forecasting-Accuracy
This repository is my solition for kaggle M5 Forecasting Accuracy. See [competition website](https://www.kaggle.com/c/m5-forecasting-accuracy) for the details.

### Result
- Local cv score: 0.56874
- Private LB score: xxxx

### Features
- Basic stat features: 7~28days rolling stat features.
- Lag features: 0~28days lag.
- Price features: rolling stat, momentum etc.
- Calendar features: seasonality, event flag, holiday flag etc.
- Other features: days from release, zero sales ratio (7~28days).

### Cross Validation
- Custom timeseries split: Last 2 month (fold1,2) + 1 year before (fold3)
- Group k-fold: 5fold, group = year + month

### Models
LightGBM with tweedie loss. Prepared 4 models with different cv strategies (TS or group) and data split (by category or by week).
Blend them by ridge regression.

|name|cv|split|fold1|fold2|fold3|avg|weight|
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|Weekly Group|Group|Week|0.60813|0.50685|0.65035|0.58844|0.4170|
|Weekly TS|TS|Week|0.60286|0.52743|0.69063|0.60698|0.2658|
|Category Group|Group|Category|0.64256|0.48684|0.69149|0.60696|0.2335|
|No split Group|Group|-|0.63070|0.49361|0.66274|0.59568|0.1091|
|Blend|-|-|0.55548|0.50797|0.64276|0.56874|-|
