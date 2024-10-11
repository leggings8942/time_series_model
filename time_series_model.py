import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from random import Random
from scipy.ndimage import shift

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def multivariate_normal_distrubution(x, mean, cov):
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(x)

class Update_Rafael:
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, beta3=0.9999, rate=1e-3):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.beta3 = beta3
        self.rate  = rate
        self.beta1t = self.beta1
        self.beta2t = self.beta2
        self.beta3t = self.beta3
        self.m = np.array([])
        self.v = np.array([])
        self.w = np.array([])
        self.isFirst = True

    def update(self, grads):
        if self.isFirst == True:
            self.m = np.zeros(grads.shape)
            self.v = np.zeros(grads.shape)
            self.w = np.zeros(grads.shape)
            self.isFirst = False

        self.m = self.beta1 * self.m + (1 - self.beta1) * grads
        m_hat = self.m / (1 - self.beta1t)

        self.v = self.beta2 * self.v + (1 - self.beta2) * (grads ** 2)
        v_hat = self.v / (1 - self.beta2t)

        self.w = self.beta3 * self.w + (1 - self.beta3) * ((grads - m_hat) ** 2)
        w_hat = self.w / (1 - self.beta3t)
        
        self.beta1t *= self.beta1
        self.beta2t *= self.beta2
        self.beta3t *= self.beta3

        return self.alpha * np.sign(grads) * np.abs(m_hat) / np.sqrt(v_hat + 1e-8) / np.sqrt(w_hat + self.rate)



class Auto_Regressive:
    def __init__(self, tol=1e-7, max_iterate=100000, learning_rate=0.001, random_state=None) -> None:
        self.lags            = 0
        self.alpha           = np.array([], dtype=np.float64)
        self.alpha0          = np.float64(0.0)
        self.sigma           = 0
        self.tol             = tol
        self.data_num        = 0
        self.max_iterate     = max_iterate
        self.correct_alpha   = Update_Rafael(alpha=learning_rate)

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random
        
        return None

    def fit(self, train_data, lags=1, solver="normal equations") -> bool:
        if type(train_data) is pd.core.series.Series:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if train_data.ndim != 1:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise

        nobs   = len(train_data)
        x_data = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        y_data = train_data[lags:]
        
        self.lags = lags
        num, s    = x_data.shape
        if solver == "normal equations":
            #正規方程式
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha, self.alpha0 = x[0:s], x[s]
        else:
            raise

        y_pred     = self.predict(x_data)
        self.sigma = np.var(y_pred - y_data)
        self.data_num = x_data.shape[0]

        return True

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        y_pred = np.sum(self.alpha * test_data, axis=1) + self.alpha0
        
        return y_pred
    
    def log_likelihood(self, test_data) -> np.float64:
        if type(test_data) is pd.core.series.Series:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 1:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise

        nobs   = len(test_data)
        x_data = np.array([test_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data = test_data[self.lags:]

        num, _ = x_data.shape
        y_pred = self.predict(x_data)

        prob           = np.frompyfunc(normal_distribution, 3, 1)(y_data, y_pred, np.sqrt(self.sigma))
        prob           = prob.astype(float).reshape([num, 1])
        log_likelihood = np.sum(np.log(prob + 1e-32))

        return log_likelihood
    
    def model_reliability(self, test_data, ic="aic") -> np.float64:
        num, s = self.data_num, self.alpha.shape[0]
        log_likelihood = self.log_likelihood(test_data)

        inf = 0
        if ic == "aic":
            inf = -2 * log_likelihood + 2 * (s + 2)
        elif ic == "bic":
            inf = -2 * log_likelihood + (s + 2) * np.log(num)
        elif ic == "hqic":
            inf = -2 * log_likelihood + 2 * (s + 2) * np.log(np.log(num))
        else:
            raise

        return inf

    def select_order(self, train_data, maxlag=15, ic="aic", solver="normal equations", isVisible=False) -> int:
        if isVisible == True:
            print(f"AR model | {ic}", flush=True)

        model_param = []
        for lag in range(1, maxlag + 1):
            self.fit(train_data, lags=lag, solver=solver)
            rel = self.model_reliability(train_data, ic=ic)
            model_param.append([rel, lag, self.alpha, self.alpha0])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag, res_alpha, res_alpha0 = np.finfo(np.float64).max, 0, 0, 0
        for elem in model_param:
            tmp_rel, tmp_lag, tmp_alpha, tmp_alpha0 = elem
            if res_rel > tmp_rel:
                res_rel    = tmp_rel
                res_lag    = tmp_lag
                res_alpha  = tmp_alpha
                res_alpha0 = tmp_alpha0
        
        self.lags               = res_lag
        self.alpha, self.alpha0 = res_alpha, res_alpha0
        
        if isVisible == True:
            print(f"selected orders | {res_lag}", flush=True)

        return self.lags

    def stat_inf(self):
        info = {}
        info["mean"]               = self.alpha0 / (1 - np.sum(self.alpha))
        info["variance"]           = self.sigma / (1 - np.sum(np.square(self.alpha)))
        info["standard deviation"] = np.sqrt(info["variance"])

        return info


class Vector_Auto_Regressive:
    def __init__(self, train_data, tol=1e-7, max_iterate=100000, learning_rate=0.001, random_state=None) -> None:
        if type(train_data) is pd.core.frame.DataFrame:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if train_data.ndim != 2:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.train_data          = train_data
        self.lags                = 0
        self.alpha               = np.zeros([1, 1])
        self.alpha0              = np.zeros([1, 1])
        self.sigma               = np.zeros([1, 1])
        self.tol                 = tol
        self.solver              = ""
        self.data_num            = 0
        self.max_iterate         = max_iterate
        self.correct_alpha       = Update_Rafael(alpha=learning_rate)
        self.unbiased_dispersion = 0
        self.dispersion          = 0
        self.ma_inf              = np.zeros([1, 1])

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random
            
    def copy(self):
        buf = []
        buf = buf + [self.train_data.copy()]
        buf = buf + [self.lags]
        buf = buf + [self.alpha.copy()]
        buf = buf + [self.alpha0.copy()]
        buf = buf + [self.sigma.copy()]
        buf = buf + [self.tol]
        buf = buf + [self.solver]
        buf = buf + [self.data_num]
        buf = buf + [self.max_iterate]
        buf = buf + [self.correct_alpha]
        buf = buf + [self.unbiased_dispersion]
        buf = buf + [self.dispersion]
        buf = buf + [self.ma_inf.copy()]
        buf = buf + [self.random_state]
        buf = buf + [self.random]
        
        return buf
    
    def restore(self, buf):
        self.train_data          = buf[0]
        self.lags                = buf[1]
        self.alpha               = buf[2]
        self.alpha0              = buf[3]
        self.sigma               = buf[4]
        self.tol                 = buf[5]
        self.solver              = buf[6]
        self.data_num            = buf[7]
        self.max_iterate         = buf[8]
        self.correct_alpha       = buf[9]
        self.unbiased_dispersion = buf[10]
        self.dispersion          = buf[11]
        self.ma_inf              = buf[12]
        self.random_state        = buf[13]
        self.random              = buf[14]
        
        return True

    def fit(self, lags=1, offset=0, solver="normal equations") -> bool:
        tmp_train_data = self.train_data[offset:]
        nobs           = len(tmp_train_data)
        x_data         = np.array([tmp_train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        y_data         = tmp_train_data[lags:]
        
        tmp_judge = y_data - np.mean(y_data, axis=0)
        tmp_judge = np.dot(tmp_judge.T, tmp_judge)
        if np.linalg.det(tmp_judge) < 1e-16:
            raise ValueError("Contains invalid time series data.")
        
        self.lags = lags
        num, s    = x_data.shape
        if solver == "normal equations":
            #正規方程式
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha, self.alpha0 = x[0:s, :], x[s, :]
            self.alpha0 = self.alpha0.reshape([1, x.shape[1]])
        else:
            raise
        
        # なぜか、共分散行列の計算に特殊な計算方法が採用されている
        # statsmodels.tsa.vector_ar.var_model.VAR を参照のこと
        # _estimate_var関数内にて当該の記述を発見
        # どうやら不偏共分散の推定量らしい
        # math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1} Z^\prime) Y
        # この式が元になっているらしい
        # よくわからないが、この式を採用することにする
        denominator   = y_data.shape[0] - y_data.shape[1] * lags - 1
        
        y_pred        = self.predict(x_data)
        diff          = y_pred - y_data
        self.sigma    = np.dot(diff.T, diff) / denominator
        self.solver   = solver
        self.data_num = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return True

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        return y_pred
    
    def get_RSS(self) -> np.ndarray:
        nobs   = len(self.train_data)
        x_data = np.array([self.train_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data = self.train_data[self.lags:]
        
        y_pred = self.predict(x_data)

        rss = np.square(y_data - y_pred)
        rss = np.sum(rss, axis=0)
        return rss
    
    def log_likelihood(self) -> np.float64:
        # なぜか、対数尤度の計算に特殊な計算方法が採用されている
        # statsmodels.tsa.vector_ar.var_model を参照のこと
        # var_loglike関数内にて当該の記述を発見
        # どうやらこれで対数尤度を計算できるらしい
        # math:: -\left(\frac{T}{2}\right) \left(\ln\left|\Omega\right| - K\ln\left(2\pi\right) - K\right)
        # この式が元になっているらしい
        # さっぱり理解できないため、通常通りに計算することにする
        
        nobs   = len(self.train_data)
        x_data = np.array([self.train_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data = self.train_data[self.lags:]

        num, _ = y_data.shape
        y_pred = self.predict(x_data)

        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma      = self.sigma * self.unbiased_dispersion / self.dispersion
        
        prob           = [multivariate_normal_distrubution(y_data[idx, :], y_pred[idx, :], tmp_sigma) for idx in range(0, num)]
        prob           = np.array(prob).reshape([num, 1])
        log_likelihood = np.sum(np.log(prob + 1e-32))

        return log_likelihood
    
    def model_reliability(self, ic="aic") -> np.float64:
        # statsmodels.tsa.vector_ar.var_model.VARResults を参照のこと
        # info_criteria関数内にて当該の記述を発見
        # 赤池情報基準やベイズ情報基準をはじめとした情報基準が特殊な形に変形されている
        # これは、サンプル数を考慮した改良版らしい
        # これを採用することとする
        
        num = self.data_num
        k   = self.alpha.size + self.alpha0.size
        #log_likelihood = self.log_likelihood()
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion

        inf = 0
        if ic == "aic":
            #inf = -2 * log_likelihood + 2 * k
            inf = np.log(np.abs(np.linalg.det(tmp_sigma))) + 2 * k / num
        elif ic == "bic":
            #inf = -2 * log_likelihood + k * np.log(num)
            inf = np.log(np.abs(np.linalg.det(tmp_sigma))) + k * np.log(num) / num
        elif ic == "hqic":
            #inf = -2 * log_likelihood + 2 * k * np.log(np.log(num))
            inf = np.log(np.abs(np.linalg.det(tmp_sigma))) + 2 * k * np.log(np.log(num)) / num
        else:
            raise

        return inf

    def select_order(self, maxlag=15, ic="aic", solver="normal equations", isVisible=False) -> int:
        if isVisible == True:
            print(f"AR model | {ic}", flush=True)

        model_param = []
        for lag in range(1, maxlag + 1):
            self.fit(lags=lag, offset=maxlag - lag, solver=solver)
            rel = self.model_reliability(ic=ic)
            model_param.append([rel, lag])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel = tmp_rel
                res_lag = tmp_lag
        
        self.fit(lags=res_lag, offset=0, solver=solver)
        if isVisible == True:
            print(f"selected orders | {res_lag}", flush=True)

        return self.lags
    
    def stat_inf(self) -> dict:
        info = {}
        if self.lags * self.alpha.shape[1] != self.alpha.shape[0]:
            print("データの次元が一致しません")
            print("lags = ", self.lags)
            print("alpha.shape = ", self.alpha.shape)
            print("時系列データ数 = ", self.alpha.shape[1])
            raise

        tmp_alpha = []
        for lag in range(0, self.lags):
            tmp_alpha.append(self.alpha[self.alpha.shape[1] * lag:self.alpha.shape[1] * (lag + 1), :].tolist())

        tmp_alpha = np.array(tmp_alpha)
        info["mean"] = np.dot(np.linalg.inv(np.identity(self.alpha.shape[1]) - np.sum(tmp_alpha, axis=0)), self.alpha0.T)

        return info
    
    def test_causality(self, causing=0, caused=1):
        backup = self.copy()
        tmp_train_data = backup[0]
        tmp_lags       = backup[1]
        tmp_alpha      = backup[2]
        tmp_solver     = backup[6]
        tmp_data_num   = backup[7]

        self.fit(lags=tmp_lags, solver=tmp_solver)
        rss1 = self.get_RSS()[caused]

        caused = caused - 1 if causing < caused else caused
        self.train_data = np.delete(tmp_train_data, causing, axis=1)
        self.fit(lags=tmp_lags, solver=tmp_solver)
        rss0 = self.get_RSS()[caused]

        num    = tmp_train_data.shape[1]
        Fvalue = (rss0 - rss1)/num / (rss1 / (tmp_data_num - tmp_alpha.shape[0] - 1))
        pvalue = stats.chi2.sf(x=Fvalue*num, df=num)

        self.restore(backup)

        return Fvalue*num, pvalue
    
    def ma_replace(self, max=10):
        ma_inf = np.zeros([max + 1, self.train_data.shape[1], self.train_data.shape[1]])
        ma_inf[0, :, :] = np.identity(self.train_data.shape[1])
        
        x_data = ma_inf[0, :, :]
        for _ in range(1, self.lags):
            x_data = np.vstack([x_data, np.zeros([self.train_data.shape[1], self.train_data.shape[1]])])

        for idx in range(1, max + 1):
            ma_inf[idx, :, :] = np.dot(self.alpha.T, x_data)
            x_data = np.vstack([ma_inf[idx, :, :], x_data[:-self.train_data.shape[1], :]])
        
        self.ma_inf = ma_inf
        return self.ma_inf

    def irf(self, period=30, orth=False):

        if orth == True:
            L = np.linalg.cholesky(self.sigma)
            D = np.diag(np.diag(L))
            P = np.dot(L, np.linalg.inv(D))
            D = D ** 2

            irf = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
            irf[0, :, :] = np.dot(P, np.identity(self.train_data.shape[1]))
            #irf[0, :, :] = np.dot(P, np.sqrt(D))

            x_data = irf[0, :, :]
            for _ in range(1, self.lags):
                x_data = np.vstack([x_data, np.zeros([self.train_data.shape[1], self.train_data.shape[1]])])
            
            for idx in range(1, period + 1):
                #tmp = self.alpha.reshape(self.lags, self.train_data.shape[1], self.train_data.shape[1])
                #tmp = tmp.swapaxes(1,2).reshape(self.lags * self.train_data.shape[1], self.train_data.shape[1])
                #irf[idx, :, :] = np.dot(x_data, tmp)
                irf[idx, :, :] = np.dot(self.alpha.T, x_data)
                x_data = np.vstack([irf[idx, :, :], x_data[:-self.train_data.shape[1], :]])
            """irf_data = self.irf(period, orth=False)
            L = np.linalg.cholesky(self.sigma)
            irf = np.array([np.dot(coefs, L) for coefs in irf_data])"""

        else:
            irf = self.ma_replace(period)

        return irf
    
    def fevd(self, period=30):
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        L = np.linalg.cholesky(tmp_sigma)
        D = np.diag(np.diag(L))
        P = np.dot(L, np.linalg.inv(D))
        D = D ** 2
        
        fevd = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
        fevd[0, :, :] = P
        
        x_data = fevd[0, :, :]
        for _ in range(1, self.lags):
            x_data = np.vstack([x_data, np.zeros([self.train_data.shape[1], self.train_data.shape[1]])])
        
        fevd[0, :, :] = fevd[0, :, :] ** 2
        for idx in range(1, period + 1):
            fevd[idx, :, :] = np.dot(self.alpha.T, x_data)
            x_data = np.vstack([fevd[idx, :, :], x_data[:-self.train_data.shape[1], :]])
            
            fevd[idx, :, :] = fevd[idx, :, :] ** 2
        
        fevd = fevd.cumsum(axis=0)
        for idx in range(0, period + 1):
            fevd[idx, :, :] = np.dot(fevd[idx, :, :], D)
        
        for idx in range(0, period + 1):
            fevd[idx, :, :] = fevd[idx, :, :] / np.sum(fevd[idx, :, :], axis=1).reshape([self.train_data.shape[1], 1])
        """fevd = self.irf(period=period, orth=True)
        fevd = (fevd ** 2).cumsum(axis=0)
        for idx in range(0, period + 1):
            fevd[idx, :, :] = fevd[idx, :, :] / np.sum(fevd[idx, :, :], axis=1).reshape([self.train_data.shape[1], 1])"""
        
        return fevd
    

class Dickey_Fuller_Test:
    def __init__(self, test_data, regression="c") -> None:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        self.test_data           = test_data
        self.regression          = regression.lower()
        self.lags                = 1
        self.alpha               = np.zeros([1, 1])
        self.alpha0              = np.zeros([1, 1])
        self.trend_1st           = np.zeros([1, 1])
        self.trend_2nd           = np.zeros([1, 1])
        self.sigma               = np.zeros([1, 1])
        self.data_num            = 0
        self.unbiased_dispersion = 0
        self.dispersion          = 0
        self.ma_inf              = np.zeros([1, 1])
    
    def fit(self) -> bool:
        tmp_train_data = self.test_data
        nobs           = len(tmp_train_data)
        x_data         = np.array([tmp_train_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data         = tmp_train_data[self.lags:]
        
        tmp_judge = y_data - np.mean(y_data, axis=0)
        tmp_judge = np.dot(tmp_judge.T, tmp_judge)
        if np.linalg.det(tmp_judge) < 1e-16:
            raise ValueError("Contains invalid time series data.")
        
        num, s = x_data.shape
        if   self.regression == "n":  # 定数項なし&トレンドなし
            A = x_data
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha,     self.alpha0    = x[0:s, :], np.zeros([1, x.shape[1]])
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(num).reshape([num, 1])])
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = x[s+1, :], np.zeros([1, x.shape[1]])
            
        elif self.regression == "ctt":# 定数項あり&2次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(num).reshape([num, 1]) ** 2])
            b = y_data
            x = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), x[s+1, :]
            
        else:
            raise
        
        self.alpha     = self.alpha.reshape([s, x.shape[1]])
        self.alpha0    = self.alpha0.reshape([1, x.shape[1]])
        self.trend_1st = self.trend_1st.reshape([1, x.shape[1]])
        self.trend_2nd = self.trend_2nd.reshape([1, x.shape[1]])
        
        # なぜか、共分散行列の計算に特殊な計算方法が採用されている
        # statsmodels.tsa.vector_ar.var_model.VAR を参照のこと
        # _estimate_var関数内にて当該の記述を発見
        # どうやら不偏共分散の推定量らしい
        # math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1} Z^\prime) Y
        # この式が元になっているらしい
        # よくわからないが、この式を採用することにする
        denominator   = y_data.shape[0] - y_data.shape[1] * self.lags - 1
        
        y_pred        = self.predict(x_data, np.arange(num).reshape([num, 1]))
        diff          = y_pred - y_data
        self.sigma    = np.dot(diff.T, diff) / denominator
        self.data_num = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return True
    
    def predict(self, test_data, time_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(time_data) is pd.core.frame.DataFrame:
            time_data = time_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if type(time_data) is list:
            time_data = np.array(time_data)
        
        if (test_data.ndim != 2) or (time_data.ndim != 2):
            print(f"test_data dims = {test_data.ndim}")
            print(f"time_data dims = {time_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        if   self.regression == "n":  # 定数項なし&トレンドなし
            y_pred = np.dot(test_data, self.alpha)
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_1st * time_data
            
        elif self.regression == "ctt":# 定数項あり&2次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_2nd * (time_data ** 2)
        
        return y_pred

