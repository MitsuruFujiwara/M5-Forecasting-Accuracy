
import feather
import gc
import holidays
import numpy as np
import pandas as pd
import sys
import warnings

from math import ceil

from utils import save2pkl, line_notify

#===============================================================================
# preprocess sell calendar
#===============================================================================

warnings.simplefilter(action='ignore')

def main(is_eval=False):
    # load csv
    df = pd.read_csv('../input/calendar.csv')

    # to datetime
    df['date'] = pd.to_datetime(df['date'])

    # seasonality
    df['seasonality'] = np.cos(np.pi*(df['date'].dt.dayofyear/366*2-1))

    # drop string columns
    df.drop('weekday',axis=1,inplace=True)

    df['day'] = df['date'].dt.day
    df['week'] = df['date'].dt.weekofyear
    df['month'] = df['date'].dt.month
    df['year'] = df['date'].dt.year
    df['year'] = (df['year'] - df['year'].min())
    df['weekofmonth'] = df['day'].apply(lambda x: ceil(x/7))

    df['dayofweek'] = df['date'].dt.dayofweek
    df['is_weekend'] = (df['dayofweek']>=5).astype(int)

    # features holiday
    df['date'] = df['date'].apply(lambda x: x.date()) # to date

    holidays_us = []
    for y in range(2011,2017):
        for ptr in holidays.UnitedStates(years = y).items():
            holidays_us.append(ptr[0])

    holidays_ca = []
    for y in range(2011,2017):
        for ptr in holidays.UnitedStates(state='CA',years = y).items():
            holidays_ca.append(ptr[0])

    holidays_tx = []
    for y in range(2011,2017):
        for ptr in holidays.UnitedStates(state='TX',years = y).items():
            holidays_tx.append(ptr[0])

    holidays_wi = []
    for y in range(2011,2017):
        for ptr in holidays.UnitedStates(state='WI',years = y).items():
            holidays_wi.append(ptr[0])

    df['is_holiday_us'] = df['date'].apply(lambda x: 1 if x in holidays_us else 0)
    df['is_holiday_ca'] = df['date'].apply(lambda x: 1 if x in holidays_ca else 0)
    df['is_holiday_tx'] = df['date'].apply(lambda x: 1 if x in holidays_tx else 0)
    df['is_holiday_wi'] = df['date'].apply(lambda x: 1 if x in holidays_wi else 0)

    # preprocess event_name_1
    # to datetime
    df['date'] = pd.to_datetime(df['date'])

    # add ramadan end dates
    ramadan_end_dates = ['2011-8-29','2012-8-18','2013-8-7','2014-7-27','2015-7-16','2016-7-5']
    for d in ramadan_end_dates:
        df.loc[df['date']==d,'event_name_1'] = 'Ramadan ends'

    # add Pesach start dates
    pesach_start_dates = ['2011-4-18','2012-4-6','2013-3-25','2014-4-14','2015-4-3','2016-4-22']
    for d in pesach_start_dates:
        df.loc[df['date']==d,'event_name_1'] = 'Pesach Start'

    # add purim start dates
    purim_start_dates = ['2011-3-19','2012-3-7','2013-2-23','2014-3-15','2015-3-4','2016-3-23']
    for d in purim_start_dates:
        df.loc[df['date']==d,'event_name_1'] = 'Purim Start'

    # add chanukah start dates
    chanukah_start_dates = ['2011-12-21','2012-12-9','2013-11-28','2014-12-17','2015-12-7','2016-12-25']
    for d in chanukah_start_dates:
        df.loc[df['date']==d,'event_name_1'] = 'Chanukah Start'

    # add isin features
    is_nba_final = []
    is_lent = []
    is_ramadan = []
    is_pesach = []
    is_purim = []
    is_chanukah = []

    tmp_nba = 0
    tmp_lent = 0
    tmp_ramadan = 0
    tmp_pesach = 0
    tmp_purim = 0
    tmp_chanukah = 0

    for e in df['event_name_1']:
        if e == 'NBAFinalsStart':
            tmp_nba = 1
        is_nba_final.append(tmp_nba)
        if e == 'NBAFinalsEnd':
            tmp_nba = 0

        if e == 'LentStart':
            tmp_lent = 1
        is_lent.append(tmp_lent)
        if e == 'Easter':
            tmp_lent = 0

        if e == 'Ramadan starts':
            tmp_ramadan = 1
        is_ramadan.append(tmp_ramadan)
        if e == 'Ramadan ends':
            tmp_ramadan = 0

        if e == 'Pesach Start':
            tmp_pesach = 1
        is_pesach.append(tmp_pesach)
        if e == 'Pesach End':
            tmp_pesach = 0

        if e == 'Purim Start':
            tmp_purim = 1
        is_purim.append(tmp_purim)
        if e == 'Purim End':
            tmp_purim = 0

        if e == 'Chanukah Start':
            tmp_chanukah = 1
        is_chanukah.append(tmp_chanukah)
        if e == 'Chanukah End':
            tmp_chanukah = 0

    df['is_NBA_final'] = is_nba_final
    df['is_lent'] = is_lent
    df['is_ramadan'] = is_ramadan
    df['is_pesach'] = is_pesach
    df['is_purim'] = is_purim
    df['is_chanukah'] = is_chanukah

    # add blackfriday flag
    blackfriday_dates = ['2011-11-25','2012-11-23','2013-11-29','2014-11-28','2015-11-27']
    df['is_blackfriday']=0
    for d in blackfriday_dates:
        df.loc[df['date']==d,'is_blackfriday'] = 1

    # factorize numerical columns
    cols_string = ['event_name_1','event_type_1','event_name_2','event_type_2']
    for c in cols_string:
        df[c], _ = pd.factorize(df[c])
        df[c].replace(-1,np.nan,inplace=True)

    # save pkl
    save2pkl('../feats/calendar.pkl', df)

    # LINE notify
    line_notify('{} done.'.format(sys.argv[0]))

if __name__ == '__main__':
    main()
