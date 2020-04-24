import pandas as pd
import numpy as np
import math
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


def fourier_sin_encoding(series, max_val, n=1):
    return np.sin((series / max_val) * (2 * math.pi * n))


def fourier_cos_encoding(series, max_val, n=1):
    return np.cos((series / max_val) * (2 * math.pi * n))


class DateFourier(BaseEstimator, TransformerMixin):
    """ A transformer that adds fourier terms for a given date period.

    For each passed in date column this transformer will compute
    a number of columns for the fourier series based on the passed in
    periods.

    Parameters
    ----------
    cols : List[str], default=None
        A list of the columns that should be transformed. If it is None
        then all detected datetime columns will be transformed.

    periods: List[{'dayfoweek', 'dayofmonth', 'dayofyear', 'monthofyear'}],
             default=['dayofyear']
        List of the different periods we want to capture with fourier terms.

    fourier : int > 0
        Integer representing how many terms of the fourier series
        should be added.

    drop_base_cols: boolean, default=True
        if true the base datetime columns will be dropped.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self,
                 cols=None,
                 fourier=1,
                 periods=['dayofyear'],
                 drop_base_cols=True):
        self.cols = cols
        self.fourier = fourier
        self.periods = periods
        self.drop_base_cols = drop_base_cols

        # check to make sure periods are valid
        valid_periods = [
            'dayofweek',
            'dayofmonth',
            'dayofyear',
            'monthofyear',
            # 'minuteofday',  # TODO add functionality for time of day
        ]
        if not all([x in valid_periods for x in self.periods]):
            raise ValueError(
                'Periods must be a subset of {}'.format(valid_periods))

    def fit(self, X, y=None):
        """Check data and if cols=None select all datetime columns.
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape (n_samples, n_features)
            The training input samples.
        y : None
            There is no need of a target in a transformer, yet the pipeline API
            requires this parameter.
        Returns
        -------
        self : object
            Returns self.
        """
        # Check input
        #X = check_array(X, accept_sparse=True, dtype=None)

        self.n_features_ = X.shape[1]

        if type(X) != pd.DataFrame:
            raise TypeError('This Transformer currently only accepts pandas \
                dataframes as inputs')

        # If no columns are passed we identify any
        # datetime columns for transformation
        if self.cols is None:
            self.cols = X.select_dtypes(
                include=[np.datetime64]).columns.tolist()
        return self

    def transform(self, X):
        """ Add Fourier series columns to datetframe.
        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        #X = check_array(X, accept_sparse=True, dtype=None)
        df = X

        for col in self.cols:
            cyclical_cols = {}
            if 'dayofweek' in self.periods:
                df['{}_day_of_week'.format(col)] = df[col].dt.dayofweek
                cyclical_cols['{}_day_of_week'.format(col)] = 7
            if 'dayofmonth' in self.periods:
                df['{}_day_of_month'.format(col)] = df[col].dt.day
                cyclical_cols['{}_day_of_month'.format(col)] = 31
            if 'dayofyear' in self.periods:
                df['{}_day_of_year'.format(col)] = df[col].dt.dayofyear
                cyclical_cols['{}_day_of_year'.format(col)] = 365
            if 'monthofyear' in self.periods:
                df['{}_month'.format(col)] = df[col].dt.month
                cyclical_cols['{}_month'.format(col)] = 12

            for fcol, max_val in cyclical_cols.items():
                for i in range(1, self.fourier + 1):
                    df[fcol + '_fourier_sin_' +
                        str(i)] = fourier_sin_encoding(df[fcol], max_val, n=i)
                    df[fcol + '_fourier_cos_' +
                        str(i)] = fourier_cos_encoding(df[fcol], max_val, n=i)

            df = df.drop(list(cyclical_cols.keys()), axis='columns')
        if self.drop_base_cols:
            df = df.drop(self.cols, axis='columns')
        return df
