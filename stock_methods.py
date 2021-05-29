import requests
import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import calendar
import time
from datetime import datetime, date, timedelta
from polygon import RESTClient
import talib
from polygon_keys import polygon_key


def get_data_by_hour(symbol):
    key = polygon_key
    client = RESTClient(key)
    query = True
    from_date = "2012-01-01"
    to_date = (datetime.today().date() +
               timedelta(days=1)).strftime("%Y-%m-%d")
    now_date = datetime.today().date().strftime("%Y-%m-%d")
    data_for_df = []
    while query:
        print(from_date)
        resp = client.stocks_equities_aggregates(
            symbol, 1, "hour", from_date, to_date, unadjusted=True, limit=50000)
        if 'results' in dir(resp):
            results = resp.results
            data_for_df = data_for_df + results[:-1]
            date = datetime.fromtimestamp((resp.results[-1]['t'])/1000.0)
            from_date = date.date().strftime("%Y-%m-%d")
            if from_date == now_date:
                query = False
        else:
            query = False

    handledData = pd.DataFrame(data=data_for_df)
    handledData.columns = ['volume', 'vw', 'open',
                           'close', 'high', 'low', 'time', 'number']
    handledData.drop_duplicates(subset=['time'], inplace=True)
    handledData.index = pd.to_datetime(
        handledData['time'].values, unit='ms').to_pydatetime()
    handledData.drop('time', axis=1, inplace=True)
    handledData.dropna(inplace=True)

    plt.figure(figsize=(15, 5))
    plt.plot(handledData.index, handledData['close'])
    plt.grid(True)
    return handledData


def get_Daily_Volatility(close, span0=24):
    # simple percentage returns
    df0 = close.pct_change()
    # 20 days, a month EWM's std as boundary
    df0 = df0.ewm(span=span0).std()
    df0.dropna(inplace=True)
    return df0


def get_3_barriers(daily_volatility, t_final, upper_lower_multipliers, prices):
    # create a container
    barriers = pd.DataFrame(columns=['days_passed', 'price', 'vert_barrier',
                            'top_barrier', 'bottom_barrier'], index=daily_volatility.index)
    for day, vol in daily_volatility.iteritems():
        days_passed = len(daily_volatility.loc[daily_volatility.index[0]: day])
        # set the vertical barrier
        if (days_passed + t_final < len(daily_volatility.index)
                and t_final != 0):
            vert_barrier = daily_volatility.index[
                days_passed + t_final]
        else:
            vert_barrier = np.nan
        # set the top barrier
        if upper_lower_multipliers[0] > 0:
            top_barrier = prices.loc[day] + prices.loc[day] * \
                upper_lower_multipliers[0] * vol
        else:
            # set it to NaNs
            top_barrier = pd.Series(index=prices.index)
        # set the bottom barrier
        if upper_lower_multipliers[1] > 0:
            bottom_barrier = prices.loc[day] - prices.loc[day] * \
                upper_lower_multipliers[1] * vol
        else:
            # set it to NaNs
            bottom_barrier = pd.Series(index=prices.index)
        barriers.loc[day, ['days_passed', 'price',
                           'vert_barrier', 'top_barrier', 'bottom_barrier']] = \
            days_passed, prices.loc[day], vert_barrier, \
            top_barrier, bottom_barrier
    barriers['out'] = None
    return barriers


def get_labels(barriers):
    for i in range(len(barriers.index)):
        start = barriers.index[i]
        end = barriers.vert_barrier[i]
        if pd.notna(end):
            # assign the initial and final price
            price_initial = barriers.price[start]
            price_final = barriers.price[end]
    # assign the top and bottom barriers
            top_barrier = barriers.top_barrier[i]
            bottom_barrier = barriers.bottom_barrier[i]
    # set the profit taking and stop loss conditons
            condition_pt = (barriers.price[start: end] >=
                            top_barrier).any()
            condition_sl = (barriers.price[start: end] <=
                            bottom_barrier).any()
    # assign the labels
            if condition_pt:
                barriers['out'][i] = 1
            elif condition_sl:
                barriers['out'][i] = -1
            else:
                barriers['out'][i] = 0
    return barriers


def create_HLCV(i, data_stock):
    df = pd.DataFrame(index=data_stock.index)
    df[f'high_{i}'] = data_stock.high.rolling(i).max()
    df[f'low_{i}'] = data_stock.low.rolling(i).min()
    df[f'close_{i}'] = data_stock.close.rolling(i).\
        apply(lambda x: x[-1])
    df[f'volume_{i}'] = data_stock.volume.rolling(i).sum()
    df[f'rsi_{i}'] = talib.RSI(data_stock['close']).rolling(i).mean()

    return df


def create_features(i, data_stock):
    df = create_HLCV(i, data_stock)
    high = df[f'high_{i}']
    low = df[f'low_{i}']
    close = df[f'close_{i}']
    volume = df[f'volume_{i}']
    # rsi = df[f'rsi_{i}']
    features = pd.DataFrame(index=data_stock.index)
    features[f'volume_{i}'] = volume
    features[f'price_spread_{i}'] = high - low
    features[f'close_loc_{i}'] = (high - close) / (high - low)
    features[f'close_change_{i}'] = close.diff()
    # features[f'rsi_{i}'] = rsi

    return features


def create_bunch_of_features(data_stock):
    days = [1, 2, 3, 5, 6, 7, 8, 9, 10, 20, 40, 60]
    bunch_of_features = pd.DataFrame(index=data_stock.index)
    for day in days:
        f = create_features(day, data_stock)
        bunch_of_features = bunch_of_features.join(f)

    # bunch_of_features['hour'] = data_stock.index.hour
    return bunch_of_features


def get_data_with_labels(symbol):
    data_stock = get_data_by_hour(symbol)
    price = data_stock['close']

    df0 = get_Daily_Volatility(price)
    daily_volatility = get_Daily_Volatility(price)
    t_final = 10
    upper_lower_multipliers = [3, 2]
    prices = price[daily_volatility.index]
    barriers = get_3_barriers(
        daily_volatility, t_final, upper_lower_multipliers, prices)
    barriers = get_labels(barriers)
    bunch_of_features = create_bunch_of_features(data_stock)

    return data_stock, barriers, bunch_of_features
