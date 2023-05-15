import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller  # ADF
from statsmodels.tsa.stattools import grangercausalitytests
from typing import Union
import statsmodels.api as sm


class Univariate_Time_Series_Analysis:

    def __init__(self, seq: Union[list, pd.Series], p_thres: float):
        self.seq = seq
        self.seq = self.seq if isinstance(self.seq, list) else self.seq.tolist()
        self.p_thres = p_thres

    def hurst(self, min_lag=2, max_lag=100):
        """Returns the Hurst Exponent of the time series vector ts"""
        lags = np.arange(min_lag, max_lag + 1)  # Create the range of lag values
        tau = [np.std(np.subtract(self.seq[lag:], self.seq[:-lag])) for lag in lags]
        params = np.polyfit(np.log10(lags), np.log10(tau), 1)
        return params, lags, tau

    def augmented_dickey_fuller_test(self):
        [t, p, c, r] = adfuller(self.seq, regression='ct', regresults=True)
        if (p <= self.p_thres) & (r.resols.pvalues[-1] <= self.p_thres):
            status = 'trend-stationary'
            is_stationary = False
        else:
            [t, p, c, r] = adfuller(self.seq, regression='c', regresults=True)
            if (p <= self.p_thres) & (r.resols.pvalues[-1] <= self.p_thres):
                status = 'drift-stationary'
                is_stationary = True
            else:
                [t, p, c, r] = adfuller(self.seq, regression='n', regresults=True)
                if p <= self.p_thres:
                    status = 'ordinary-stationary'
                    is_stationary = True
                else:
                    status = 'no-stationary'
                    is_stationary = False
        return status, is_stationary

    def integrated_order_test(self):
        status, is_stationary = self.augmented_dickey_fuller_test()
        integrated_order = 0
        while not is_stationary:
            self.seq = np.subtract(self.seq[1:], self.seq[:-1])
            status, is_stationary = self.augmented_dickey_fuller_test()
            integrated_order += 1
        return integrated_order

    def run(self):
        params, lags, tau = self.hurst()
        hurst = params[0]
        status, is_stationary = self.augmented_dickey_fuller_test()
        integrated_order = self.integrated_order_test()
        print('hurst exponent: {} (MeanReverting: 0/ BrownianMotion: 0.5/ Trending: 1)'.format(hurst))
        print('status:', status)
        print('is_stationary:', is_stationary)
        print('integrated_order:', integrated_order)
        return hurst, status, is_stationary, integrated_order


class Multivariate_Time_Series_Analysis:

    def __init__(self, seqs: pd.DataFrame, p_thres: float):
        self.seqs = seqs
        self.p_thres = p_thres

    def grangers_causation_matrix(self, maxlag=1, test_method='ssr_chi2test'):
        granger_x_list = []
        for c in self.seqs.columns:
            for r in self.seqs.columns:
                if c != r:
                    # test if c Granger case r
                    test_result = grangercausalitytests(self.seqs[[r, c]], maxlag=[maxlag], verbose=False)
                    p_value = test_result[maxlag][0][test_method][1]
                    if p_value <= self.p_thres:
                        granger_x_list.append(c)
                    else:
                        granger_x_list.append(np.NaN)

        return granger_x_list

    def cointegration_test(self):
        y_ts_analysis_instance = Univariate_Time_Series_Analysis(self.seqs.iloc[:, 0], self.p_thres)
        x_ts_analysis_instance = Univariate_Time_Series_Analysis(self.seqs.iloc[:, 1], self.p_thres)

        y_integrated_order = y_ts_analysis_instance.integrated_order_test()
        x_integrated_order = x_ts_analysis_instance.integrated_order_test()

        is_cointegrated = False

        if (min(y_integrated_order, x_integrated_order) != 0) & (y_integrated_order == x_integrated_order):
            y = self.seqs.iloc[:, 0].astype(float)
            X = sm.add_constant(self.seqs.iloc[:, 1]).astype(float)
            model = sm.OLS(endog=y, exog=X)
            result = model.fit()
            resid = result.resid
            y_ts_analysis_instance = Univariate_Time_Series_Analysis(resid, self.p_thres)
            status, is_stationary = y_ts_analysis_instance.augmented_dickey_fuller_test()
            if is_stationary:
                is_cointegrated = True

        return is_cointegrated, resid

    def run(self):
        granger_x_list = self.grangers_causation_matrix()
        is_cointegrated, resid = self.cointegration_test()
        print('granger_x:', granger_x_list)
        print('is_cointegrated:', is_cointegrated)
        return granger_x_list, is_cointegrated, resid


if __name__ == '__main__':
    rand = pd.Series(np.cumsum(np.random.randn(1000) + 0.01))
    df = pd.DataFrame([np.random.randn(1000), np.random.randn(1000)]).T
    df.columns = ['ts1', 'ts2']

    hurst, status, is_stationary, integrated_order = Univariate_Time_Series_Analysis(rand, 0.1).run()

    granger_x_list, is_cointegrated = Multivariate_Time_Series_Analysis(df, 0.1).run()
