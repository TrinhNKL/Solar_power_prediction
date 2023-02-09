from datetime import datetime
from pytz import timezone
import pytz
import pandas as pd 
from sklearn.model_selection import train_test_split

def describtive_statistics(data: pd.DataFrame):
    # replace the negative values of `Radiation` by zero
    data.loc[data.Radiation.lt(0), "Radiation"] = data.Radiation[data.Radiation < 0] * 0

    # get the statistical summary
    return data.describe()

def convert_and_add_timecol(data: pd.DataFrame):
    hawaii= timezone('Pacific/Honolulu')
    data.index =  pd.to_datetime(data['UNIXTime'], unit='s')
    data.index = data.index.tz_localize(pytz.utc).tz_convert(hawaii)
    data['MonthOfYear'] = data.index.strftime('%m').astype(int)
    data['DayOfYear'] = data.index.strftime('%j').astype(int)
    data['WeekOfYear'] = data.index.strftime('%U').astype(int)
    data['TimeOfDay(h)'] = data.index.hour
    data['TimeOfDay(m)'] = data.index.hour*60 + data.index.minute
    data['TimeOfDay(s)'] = data.index.hour*60*60 + data.index.minute*60 + data.index.second
    data['TimeSunRise'] = pd.to_datetime(data['TimeSunRise'], format='%H:%M:%S')
    data['TimeSunSet'] = pd.to_datetime(data['TimeSunSet'], format='%H:%M:%S')
    data['DayLength(s)'] = data['TimeSunSet'].dt.hour*60*60 \
                           + data['TimeSunSet'].dt.minute*60 \
                           + data['TimeSunSet'].dt.second \
                           - data['TimeSunRise'].dt.hour*60*60 \
                           - data['TimeSunRise'].dt.minute*60 \
                           - data['TimeSunRise'].dt.second
    data.drop(['Data','Time','TimeSunRise','TimeSunSet'], inplace=True, axis=1)
    
    return data

def agg_data(data:pd.DataFrame):
    grouped_m=data.groupby('MonthOfYear').mean().reset_index()
    grouped_w=data.groupby('WeekOfYear').mean().reset_index()
    grouped_d=data.groupby('DayOfYear').mean().reset_index()
    grouped_h=data.groupby('TimeOfDay(h)').mean().reset_index()
    return grouped_d, grouped_m, grouped_d, grouped_h

def data_split(X, y):   
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=10)
    return X_train, X_test, y_train, y_test