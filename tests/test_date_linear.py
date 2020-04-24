import pytest
import pandas as pd
import datetime as dt

from skautobots.date_processing import DateLinear


@pytest.fixture
def data():
    start = dt.datetime(year=2015, month=1, day=1)
    records = []
    for x in range(1000):
        sample_dt = start + dt.timedelta(days=x)
        records.append({
            'sample_datetime': sample_dt,
            'sample_date': sample_dt.date,
            'sample_time': sample_dt.time
        })
    df = pd.DataFrame.from_records(records)
    return df


def test_datetime_column_runs(data):
    df = data
    transformer = DateLinear(
        cols=['sample_datetime'],
        periods=[
            'year',
            'dayofweek',
            'dayofmonth',
            'dayofyear',
            'monthofyear'
        ]
    )
    transformer.fit(df)
    transformed_df = transformer.transform(df)
    assert True
    
