Files for the Kaggle competition [Store Sales - Time Series Forecasting](https://www.kaggle.com/competitions/store-sales-time-series-forecasting/).

## competition goals

The goal of the competition is to forecast multiple time-series of store sales for a period of 16 days. The data contains the following:
- 1782 time-series = 54 stores x 33 families 
- 1688 days of training data (one observation per day)
- additional covariates: number of items on promotion, holiday events

## project description

The project contains two notebooks which explore forecasting a subset of the time-series using [Prophet](https://github.com/facebook/prophet) and [Chronos](https://github.com/amazon-science/chronos-forecasting) respectively.

The script `preprocessing.py` should be ran first and will process the competition data into the format used afterwards.

## submission

The submission is created by running the `forecast.py` script.
The score is 0.387 (RMSLE), which (at time of submission) places at rank 41 of the rolling leaderboard.
