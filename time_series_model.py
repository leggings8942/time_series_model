import numpy as np
import scipy as sp
import pandas as pd
from scipy import stats
from sklearn.linear_model import ElasticNet

def modified_cholesky(x):
    if type(x) is list:
        x = np.array(x)
        
    if x.ndim != 2:
        print(f"x dims = {x.ndim}")
        raise ValueError("エラー：：次元数が一致しません。")
    
    if x.shape[0] != x.shape[1]:
        print(f"x shape = {x.shape}")
        raise ValueError("エラー：：正方行列ではありません。")
    
    n = x.shape[0]
    d = np.diag(x).copy()
    L = np.tril(x, k=-1).copy() + np.identity(n)
    
    for idx1 in range(1, n):
        prev = idx1 - 1
        tmp  = d[0:prev] if d[0:prev].size != 0 else 0
        tmp  = np.dot(L[idx1:, 0:prev], (L[prev, 0:prev] * tmp).T)
        
        DIV  = d[prev] if d[prev] != 0 else 1e-16
        L[idx1:, prev] = (L[idx1:, prev] - tmp) / DIV
        d[idx1]       -= np.sum((L[idx1, 0:idx1] ** 2) * d[0:idx1])
    
    d = np.diag(d)
    return L, d

def normal_distribution(x, loc=0, scale=1):
    return stats.norm.pdf(x, loc=loc, scale=scale)

def multivariate_normal_distrubution(x, mean, cov):
    return stats.multivariate_normal(mean=mean, cov=cov, allow_singular=True).pdf(x)

# 軟判別閾値関数
def soft_threshold(x, α):
    return np.sign(x) * np.maximum(np.abs(x) - α, 0)

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
        self.learn_flg       = False

        self.random_state = random_state
        if random_state != None:
            self.random = np.random
            self.random.seed(seed=self.random_state)
        else:
            self.random = np.random
        
        return None

    def fit(self, train_data, lags=1, solver="normal equations") -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        
        if type(train_data) is pd.core.series.Series:
            train_data = train_data.to_numpy()
        
        if type(train_data) is list:
            train_data = np.array(train_data)
        
        if train_data.ndim != 1:
            print(f"train_data dims = {train_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise

        nobs = len(train_data)
        if not nobs - lags - train_data.shape[1] * lags > 0:
            # データ数に対して、最尤推定対象が多すぎる
            self.learn_flg = False
            return self.learn_flg
        
        x_data = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        y_data = train_data[lags:]
        
        self.lags = lags
        num, s    = x_data.shape
        if solver == "normal equations":
            #正規方程式
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha, self.alpha0 = x[0:s], x[s]
        else:
            raise
        
        self.learn_flg = True
        y_pred         = self.predict(x_data)
        self.sigma     = np.var(y_pred - y_data)
        self.data_num  = x_data.shape[0]

        return self.learn_flg

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
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

        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
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
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
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
        
        nobs = len(train_data)
        if nobs <= maxlag:
            maxlag = nobs - 1

        model_param = []
        for lag in range(1, maxlag + 1):
            flg = self.fit(train_data, lags=lag, solver=solver)
            
            if flg:
                rel = self.model_reliability(train_data, ic=ic)
                model_param.append([rel, lag])
            else:
                rel = np.finfo(np.float64).max
                model_param.append([rel, lag])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel    = tmp_rel
                res_lag    = tmp_lag
        
        res_lag = res_lag if res_lag != 0 else 1
        self.fit(train_data, lags=res_lag, solver=solver)
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了しませんでした。")
            raise
        
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
        self.learn_flg           = False

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
        buf = buf + [self.learn_flg]
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
        self.learn_flg           = buf[13]
        self.random_state        = buf[14]
        self.random              = buf[15]
        
        return True

    def fit(self, lags=1, offset=0, solver="normal equations") -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        
        tmp_train_data = self.train_data[offset:]
        nobs           = len(tmp_train_data)
        
        if not nobs - lags - tmp_train_data.shape[1] * lags - 1 > 0:
            # データ数に対して、最尤推定対象が多すぎる
            self.learn_flg = False
            return self.learn_flg
        
        x_data         = np.array([tmp_train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        y_data         = tmp_train_data[lags:]
        
        self.lags = lags
        num, s    = x_data.shape
        if solver == "normal equations":
            #正規方程式
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
                
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
        
        self.learn_flg = True
        y_pred         = self.predict(x_data)
        diff           = y_pred - y_data
        self.sigma     = np.dot(diff.T, diff) / denominator
        self.solver    = solver
        self.data_num  = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return self.learn_flg

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
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
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
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
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        num = self.data_num
        k   = self.alpha.size + self.alpha0.size
        #log_likelihood = self.log_likelihood()
        
        # caution!!!
        # 本ライブラリでは、データ数に対して最尤推定対象が多い場合にもできる限り処理を続けるように調整してある
        # しかし、この場合に分散共分散行列の正定値性が保てなくなるという問題が発生する
        # また、入力された時系列データ自体に誤りが存在する場合にも正定値性が保てなくなる
        # 正定値行列でない場合には対数尤度の計算ができなくなる
        # この問題の対策のために対数尤度の近似値を求める処理に変更していることに注意
        # 参考URL:
        # https://seetheworld1992.hatenablog.com/entry/2017/03/22/194932
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        det_sigma = np.linalg.det(tmp_sigma)
        det_sigma = det_sigma if det_sigma != 0 else 1e-16

        inf = 0
        if ic == "aic":
            #inf = -2 * log_likelihood + 2 * k
            inf = np.log(np.abs(det_sigma)) + 2 * k / num
        elif ic == "bic":
            #inf = -2 * log_likelihood + k * np.log(num)
            inf = np.log(np.abs(det_sigma)) + k * np.log(num) / num
        elif ic == "hqic":
            #inf = -2 * log_likelihood + 2 * k * np.log(np.log(num))
            inf = np.log(np.abs(det_sigma)) + 2 * k * np.log(np.log(num)) / num
        else:
            raise

        return inf

    def select_order(self, maxlag=15, ic="aic", solver="normal equations", isVisible=False) -> int:
        if isVisible == True:
            print(f"AR model | {ic}", flush=True)
        
        nobs = len(self.train_data)
        if nobs <= maxlag:
            maxlag = nobs - 1

        model_param = []
        for lag in range(1, maxlag + 1):
            flg = self.fit(lags=lag, offset=maxlag - lag, solver=solver)
            
            if flg:
                rel = self.model_reliability(ic=ic)
                model_param.append([rel, lag])
            else:
                rel = np.finfo(np.float64).max
                model_param.append([rel, lag])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel = tmp_rel
                res_lag = tmp_lag
        
        res_lag = res_lag if res_lag != 0 else 1
        self.fit(lags=res_lag, offset=0, solver=solver)
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了しませんでした。")
            raise
        
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

    def irf(self, period=30, orth=False, isStdDevShock=True):
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise

        if orth == True:
            A, D = modified_cholesky(self.sigma)

            irf = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
            if isStdDevShock:
                irf[0, :, :] = np.dot(A, np.sqrt(D))
            else:
                irf[0, :, :] = np.dot(A, np.identity(self.train_data.shape[1]))

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
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        A, D = modified_cholesky(tmp_sigma)
        
        fevd = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
        fevd[0, :, :] = A
        
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


class Sparse_Vector_Auto_Regressive:
    def __init__(self,
                 train_data,                         # 学習対象時系列データ
                 norm_α:float=1.0,                   # L1・L2正則化パラメータの強さ
                 l1_ratio:float=0.01,                  # L1・L2正則化の強さ配分・比率
                 tol:float=1e-6,                     # 許容誤差
                 isStandardization:bool=True,        # 正規化処理の適用有無
                 max_iterate:int=30000,              # 最大ループ回数
                 learning_rate:float=0.0001,         # 学習係数
                 random_state=None) -> None:         # 乱数のシード値
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
        self.norm_α              = np.abs(norm_α)
        self.l1_ratio            = np.where(l1_ratio < 0, 0, np.where(l1_ratio > 1, 1, l1_ratio))
        self.isStandardization   = isStandardization
        self.x_standardization   = np.empty([2, 1])
        self.y_standardization   = np.empty([2, 1])
        self.tol                 = tol
        self.solver              = ""
        self.data_num            = 0
        self.max_iterate         = max_iterate
        self.correct_alpha       = Update_Rafael(alpha=learning_rate)
        self.correct_alpha0      = Update_Rafael(alpha=learning_rate)
        self.unbiased_dispersion = 0
        self.dispersion          = 0
        self.ma_inf              = np.zeros([1, 1])
        self.learn_flg           = False

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
        buf = buf + [self.norm_α]
        buf = buf + [self.l1_ratio]
        buf = buf + [self.isStandardization]
        buf = buf + [self.x_standardization]
        buf = buf + [self.y_standardization]
        buf = buf + [self.tol]
        buf = buf + [self.solver]
        buf = buf + [self.data_num]
        buf = buf + [self.max_iterate]
        buf = buf + [self.correct_alpha]
        buf = buf + [self.correct_alpha0]
        buf = buf + [self.unbiased_dispersion]
        buf = buf + [self.dispersion]
        buf = buf + [self.ma_inf.copy()]
        buf = buf + [self.learn_flg]
        buf = buf + [self.random_state]
        buf = buf + [self.random]
        
        return buf
    
    def restore(self, buf):
        self.train_data          = buf[0]
        self.lags                = buf[1]
        self.alpha               = buf[2]
        self.alpha0              = buf[3]
        self.sigma               = buf[4]
        self.norm_α              = buf[5]
        self.l1_ratio            = buf[6]
        self.isStandardization   = buf[7]
        self.x_standardization   = buf[8]
        self.y_standardization   = buf[9]
        self.tol                 = buf[10]
        self.solver              = buf[11]
        self.data_num            = buf[12]
        self.max_iterate         = buf[13]
        self.correct_alpha       = buf[14]
        self.correct_alpha0      = buf[15]
        self.unbiased_dispersion = buf[16]
        self.dispersion          = buf[17]
        self.ma_inf              = buf[18]
        self.learn_flg           = buf[19]
        self.random_state        = buf[20]
        self.random              = buf[21]
        
        return True

    def fit(self, lags:int=1, offset:int=0, solver:str='external library', visible_flg:bool=False) -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        # また、solverとしてISTAを使用する際にも注意が必要である
        # ISTAは勾配降下法に似た特徴を有しており、対象の最適化パラメータのスケールに弱い
        # 最適化対象のパラメータの解析解のスケールに依存して、必要な更新回数が多くなる
        # スケールが極端に大きい場合などには事実上収束しないが、そもそも解析解のスケールを事前に知らない・気にしていない場合も多い
        # そのような場合には、教師データ(X, Y)をそれぞれ正規化することで対処できる
        # isStandardization=True に設定しておくことを強く推奨する
        
        if len(self.train_data) <= offset:
            # データ数に対して、オフセットが大き過ぎる
            self.learn_flg = False
            return self.learn_flg
        
        tmp_train_data = self.train_data[offset:]
        nobs           = len(tmp_train_data)
        
        if nobs <= lags:
            # 学習対象データ数に対して、ラグが大き過ぎる
            self.learn_flg = False
            return self.learn_flg
        
        x_data = np.array([tmp_train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        y_data = tmp_train_data[lags:]
        
        # 正規化指定の有無
        if self.isStandardization:
            # x軸の正規化
            _, s = x_data.shape
            self.x_standardization    = np.empty([2, s])
            self.x_standardization[0] = np.mean(x_data, axis=0)
            self.x_standardization[1] = np.std( x_data, axis=0)

            # 標準偏差が0の場合
            zero_judge = (self.x_standardization[1] == 0)
            self.x_standardization[0][zero_judge] = 0
            self.x_standardization[1][zero_judge] = 1

            x_data = (x_data - self.x_standardization[0]) / self.x_standardization[1]
            
            # y軸の正規化
            _, s = y_data.shape
            self.y_standardization    = np.empty([2, s])
            self.y_standardization[0] = np.mean(y_data, axis=0)
            self.y_standardization[1] = np.std( y_data, axis=0)

            # 標準偏差が0の場合
            zero_judge = (self.y_standardization[1] == 0)
            self.y_standardization[0][zero_judge] = 0
            self.y_standardization[1][zero_judge] = 1

            y_data = (y_data - self.y_standardization[0]) / self.y_standardization[1]
        
        # 本ライブラリで実装されているアルゴリズムは以下の4点となる
        # ・sklearnライブラリに実装されているElasticNet(外部ライブラリ)
        # ・座標降下法アルゴリズム(CD: Coordinate Descent Algorithm)
        # ・メジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)の亜種
        # これらのアルゴリズムは全て同じ目的関数を最適化している
        # しかし、実際に同一のパラメータでパラメータ探索をさせても同一の解は得られない
        # これは、実装の細かな違いによるものであったり、解析解ではなく近似解が得られるためであったりする
        # 特にISTAは勾配降下法と同等の性質を有しているため、異なる近似解が得られる
        # すなわち実行のたびに異なる解が導かれるかつ極所最適解に落ち着くことがある
        # また、外部ライブラリとしてsklearn.linear_model.ElasticNetを利用することもできる
        # この外部ライブラリは内部で座標降下法で探索を行っている点で本ライブラリと同等である
        # 一方で、この外部ライブラリはC言語(Cython)を利用してチューニングが行われている
        # また広く公開され、多くの人に利用されているライブラリでもあるため速度・品質ともにレベルが高い
        # 探索解の品質を保証したいのであれば、外部ライブラリの利用を強く推奨する
        # 最後に広く認められているわけではないため使用の際には注意が必要であるが、本ライブラリにて実装済みの
        # これら3種類のアルゴリズムが想定する目的関数は以下のとおり
        # A = 説明変数x + 切片b の行列(データ数n ✖️ (説明変数数s + 1))
        # B = 目的変数y の行列(データ数n ✖️ 目的変数数m)
        # X = 説明変数xの係数 + 切片bの係数 の行列((説明変数数s + 1) ✖️ 目的変数数m)
        # λ_1 = 正則化の強度 * l1_ratio
        # λ_2 = 正則化の強度 * (1 - l1_ratio)
        # math: \begin{equation}
        # math: \begin{split}
        # math: Objective &= \frac{1}{n} \| B - AX \|_2^2 + \frac{λ_2}{2} \| X \|_2^2 + λ_1 \|X\|_1 \\
        # math: &= tr [ \left( B - AX \right) ^T \left( B - AX \right) ] + \frac{λ_2 n}{2} tr [ X^T X ] + λ_1 n \sum_{i=1} |x_i |
        # math: \end{split}
        # math: \end{equation}
        # 参考までに各オプションごとの実行速度は以下の通り
        # external library  >>  coordinate descent  >>  ISTA
        
        self.lags = lags
        num, s    = x_data.shape
        if   solver == "external library":
            model = ElasticNet(alpha=self.norm_α, l1_ratio=self.l1_ratio.tolist(), max_iter=self.max_iterate, tol=self.tol)
            model.fit(x_data, y_data)
            
            self.alpha, self.alpha0 = model.coef_.T, model.intercept_
            self.alpha0 = self.alpha0.reshape([1, y_data.shape[1]])
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * num
                A       = np.hstack([x_data, np.ones([num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                ABSO = np.abs(X)
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(s + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        elif solver == "coordinate descent":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # 注意点として、切片に対してはラッソ最適化を行わないことが挙げられる
            # リッジ最適化は一般に係数を0にするためではなく、最適化対象のパラメータ全体を小さく保つために利用される
            # 一方で、ラッソ最適化は係数を0にするために利用される手法である
            # そのため、一般にはラッソ最適化を切片に対しては適用しない習慣がある
            # このライブラリもこの習慣に従うことにする
            # 実装アルゴリズムは座標降下法である
            # できる限り高速に処理を行いたかったので、このような実装になった
            # このアルゴリズムの計算量は、O(ループ回数 × 説明変数の数 × O(行列積))である
            # 1×M, M×Lの大きさを持つ行列A, Bを想定すると、行列積の計算量はO(ML)となる
            # このSVARライブラリではそれぞれ、M=(説明変数の数 + 1) L=目的変数の数に対応している
            # 計算量オーダーを書き直すと O(ループ回数 × 説明変数の数 × ML)となる
            # このアルゴリズムを利用するにあたって、学習対象データの正規化などの条件は特にない
            l1_norm = self.norm_α * self.l1_ratio       * num
            l2_norm = self.norm_α * (1 - self.l1_ratio) * num
            A       = np.hstack([x_data, np.ones([num, 1])])
            b       = y_data
            
            L = np.dot(A.T, A) + l2_norm * np.identity(s + 1)
            R = np.dot(A.T, b)
            D = np.diag(np.diag(L))
            G = np.diag(L)
            C = L - D
                
            # 切片に対して、L1正則化を適用しない
            C[s, :] = 0
            
            x_new = np.zeros([s + 1, b.shape[1]])
            x_new[s, :] = R[s, :] / G[s]
            for idx1 in range(0, self.max_iterate):
                x_old = x_new.copy()

                for idx3 in range(0, s):
                    tmp = R[idx3, :] - np.dot(C[idx3, :], x_new)
                    x_new[idx3, :] = soft_threshold(tmp, l1_norm) / G[idx3]
                
                ΔDiff = np.sqrt(np.sum((x_new - x_old) ** 2))
                if visible_flg:
                    mse = np.sum(ΔDiff ** 2) / num
                    print(f"ite:{idx1+1}  mse:{mse}  ΔDiff:{ΔDiff}")
                
                if ΔDiff <= self.tol:
                    break
            
            x = x_new
            self.alpha, self.alpha0 = x[0:s, :], x[s, :]
            self.alpha0 = self.alpha0.reshape([1, x.shape[1]])
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * num
                A       = np.hstack([x_data, np.ones([num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                ABSO = np.abs(X)
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(s + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        elif solver == "ISTA":
            # ラッソ最適化(L1正則化)とリッジ最適化(L2正則化)を行なっている
            # 注意点として、切片に対してはラッソ最適化を行わないことが挙げられる
            # リッジ最適化は一般に係数を0にするためではなく、最適化対象のパラメータ全体を小さく保つために利用される
            # 一方で、ラッソ最適化は係数を0にするために利用される手法である
            # そのため、一般にはラッソ最適化を切片に対しては適用しない習慣がある
            # このライブラリもこの習慣に従うことにする
            # 実装アルゴリズムは一般的なメジャライザー最適化(ISTA: Iterative Shrinkage soft-Thresholding Algorithm)である
            # このアルゴリズムを利用する際の注意点として、以下の２つが挙げられる
            # ・教師データ(X, Y)がそれぞれ正規化されている必要があること
            # ・大域的最適解を探索できるとは限らないこと
            # 正規化されていない場合にはうまく収束しないくなる等、アルゴリズムが機能しなくなる可能性がある
            # isStandardization=True に設定しておけば、問題ない
            
            l1_norm     = self.norm_α * self.l1_ratio       * num
            l2_norm     = self.norm_α * (1 - self.l1_ratio) * num
            self.alpha  = self.random.random([s, y_data.shape[1]])
            self.alpha0 = self.random.random([1, y_data.shape[1]])
            for idx in range(0, self.max_iterate):
                y_pred  = np.dot(x_data, self.alpha) + self.alpha0
                
                ΔLoss   = y_data - y_pred
                Δalpha  = np.dot(x_data.T, ΔLoss) - l2_norm * self.alpha
                Δalpha0 = np.sum(ΔLoss, axis=0)   - l2_norm * self.alpha0
                
                diff_alpha  = self.correct_alpha.update(Δalpha)
                diff_alpha0 = self.correct_alpha0.update(Δalpha0)
                rho         = diff_alpha / (Δalpha + 1e-32)
                
                # 切片に対して、L1正則化を適用しない
                self.alpha  = soft_threshold(self.alpha + diff_alpha, l1_norm * rho)
                self.alpha0 = self.alpha0 + diff_alpha0
                
                update_diff = np.sqrt(np.sum(Δalpha ** 2) + np.sum(Δalpha0 ** 2))
                if visible_flg and (idx % 5000 == 0):
                    mse = np.sum(ΔLoss ** 2) / num
                    print(f"ite:{idx+1}  mse:{mse}  update_diff:{update_diff}")
                
                if update_diff <= self.tol:
                    break
            
            if visible_flg:
                l1_norm = self.norm_α * self.l1_ratio       * num
                l2_norm = self.norm_α * (1 - self.l1_ratio) * num
                A       = np.hstack([x_data, np.ones([num, 1])])
                B       = y_data
                X       = np.vstack([self.alpha, self.alpha0])
                DIFF = B - np.dot(A, X)
                DIFF = np.dot(DIFF.T, DIFF)
                SQUA = np.dot(X.T, X)
                ABSO = np.abs(X)
                OBJE = 1 / 2 * np.sum(np.diag(DIFF)) + l2_norm / 2 * np.sum(np.diag(SQUA)) + l1_norm * np.sum(ABSO)
                print("平均二乗誤差(MSE):", np.sum(np.diag(DIFF)) / num, flush=True)
                print("L2正則化項(l2 norm):", np.sum(np.diag(SQUA)))
                print("L1正則化項(l1 norm):", np.sum(ABSO))
                print("目的関数(Objective): ", OBJE)
                
                DLoss = np.dot(A.T, B) - l1_norm * np.sign(X) - np.dot(np.dot(A.T, A) + l2_norm * np.identity(s + 1), X)
                print("目的関数(Objective)の微分: ", np.abs(DLoss).sum())
            
        else:
            raise
        
        # 不偏共分散行列を計算するためには、本来以下のような母数を採用する必要がある
        # 母数 = 学習データ数 - 最適化対象変数の数
        # しかし、このモデルでは”学習データ数 << 最適化対象変数の数”という状況下で利用されることを想定している
        # この状況下では、母数が負の値になってしまうため採用することができない
        # そのため、苦肉の策として 母数 = 学習データ数 - 1 を採用することにした。根拠は弱い。
        # 今後、このような状況での最適な不偏共分散行列の求め方が判明したならば積極的に変更を加えることとする
        denominator    = num - 1
        
        self.learn_flg = True
        if self.isStandardization:
            y_pred     = self.predict(x_data * self.x_standardization[1] + self.x_standardization[0])
            diff       = y_data - (y_pred - self.y_standardization[0]) / self.y_standardization[1]
        else:
            y_pred     = self.predict(x_data)
            diff       = y_data -  y_pred
            
        
        self.sigma     = np.dot(diff.T, diff) / denominator
        self.solver    = solver
        self.data_num  = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return self.learn_flg

    def predict(self, test_data) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if test_data.ndim != 2:
            print(f"test_data dims = {test_data.ndim}")
            print("エラー：：次元数が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        if self.isStandardization:
            test_data = (test_data - self.x_standardization[0]) / self.x_standardization[1]
        
        y_pred = np.dot(test_data, self.alpha) + self.alpha0
        if self.isStandardization:
            y_pred = y_pred * self.y_standardization[1] + self.y_standardization[0]
        
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
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
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
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        num = self.data_num
        k   = self.alpha.size + self.alpha0.size
        #log_likelihood = self.log_likelihood()
        
        # caution!!!
        # 本ライブラリでは、データ数に対して最尤推定対象が多い場合にもできる限り処理を続けるように調整してある
        # しかし、この場合に分散共分散行列の正定値性が保てなくなるという問題が発生する
        # また、入力された時系列データ自体に誤りが存在する場合にも正定値性が保てなくなる
        # 正定値行列でない場合には対数尤度の計算ができなくなる
        # この問題の対策のために対数尤度の近似値を求める処理に変更していることに注意
        # 参考URL:
        # https://seetheworld1992.hatenablog.com/entry/2017/03/22/194932
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        det_sigma = np.linalg.det(tmp_sigma)
        det_sigma = det_sigma if det_sigma != 0 else 1e-16

        inf = 0
        if ic == "aic":
            #inf = -2 * log_likelihood + 2 * k
            inf = np.log(np.abs(det_sigma)) + 2 * k / num
        elif ic == "bic":
            #inf = -2 * log_likelihood + k * np.log(num)
            inf = np.log(np.abs(det_sigma)) + k * np.log(num) / num
        elif ic == "hqic":
            #inf = -2 * log_likelihood + 2 * k * np.log(np.log(num))
            inf = np.log(np.abs(det_sigma)) + 2 * k * np.log(np.log(num)) / num
        else:
            raise

        return inf

    def select_order(self, maxlag=15, ic="aic", solver="external library", isVisible=False) -> int:
        if isVisible == True:
            print(f"SVAR model | {ic}", flush=True)
        
        nobs = len(self.train_data)
        if nobs <= maxlag:
            maxlag = nobs - 1

        model_param = []
        for lag in range(1, maxlag + 1):
            flg = self.fit(lags=lag, offset=maxlag - lag, solver=solver)
            
            if flg:
                rel = self.model_reliability(ic=ic)
                model_param.append([rel, lag])
            else:
                rel = np.finfo(np.float64).max
                model_param.append([rel, lag])

            if isVisible == True:
                print(f"SVAR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel = tmp_rel
                res_lag = tmp_lag
        
        res_lag = res_lag if res_lag != 0 else 1
        self.fit(lags=res_lag, offset=0, solver=solver)
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了しませんでした。")
            raise
        
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
        tmp_solver     = backup[11]
        tmp_data_num   = backup[12]

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

    def irf(self, period=30, orth=False, isStdDevShock=True):
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise

        if orth == True:
            A, D = modified_cholesky(self.sigma)

            irf = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
            if isStdDevShock:
                irf[0, :, :] = np.dot(A, np.sqrt(D))
            else:
                irf[0, :, :] = np.dot(A, np.identity(self.train_data.shape[1]))

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
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        A, D = modified_cholesky(tmp_sigma)
        
        fevd = np.zeros([period + 1, self.train_data.shape[1], self.train_data.shape[1]])
        fevd[0, :, :] = A
        
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
        
        if (test_data.ndim != 2) or (test_data.shape[1] != 1):
            print(f"test_data dims  = {test_data.ndim}")
            print(f"test_data shape = {test_data.shape}")
            print("エラー：：次元数または形が一致しません。")
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
        self.ρvalue              = 0
        self.pvalue              = 0
        self.learn_flg           = False
    
    def fit(self) -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        
        tmp_train_data = self.test_data
        nobs           = len(tmp_train_data)
        
        if not nobs - self.lags - tmp_train_data.shape[1] * self.lags - 1 > 0:
            # データ数に対して、最尤推定対象が多すぎる
            self.learn_flg = False
            return self.learn_flg
        
        x_data         = np.array([tmp_train_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
        y_data         = tmp_train_data[self.lags:]
        
        num, s = x_data.shape
        if   self.regression == "n":  # 定数項なし&トレンドなし
            A = x_data
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], np.zeros([1, x.shape[1]])
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(1, num+1).reshape([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = x[s+1, :], np.zeros([1, x.shape[1]])
            
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(1, num+1).reshape([num, 1]), np.arange(1, num+1).reshape([num, 1]) ** 2])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = x[s+1, :], x[s+2, :]
            
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
        
        self.learn_flg = True
        y_pred         = self.predict(x_data, np.arange(1, num+1).reshape([num, 1]))
        diff           = y_pred - y_data
        self.sigma     = np.dot(diff.T, diff) / denominator
        self.data_num  = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return True
    
    def predict(self, test_data, time_data = np.array([[]])) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(time_data) is pd.core.frame.DataFrame:
            time_data = time_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if type(time_data) is list:
            time_data = np.array(time_data)
        
        if (test_data.ndim != 2) or (test_data.shape[1] != 1) or (time_data.ndim != 2) or (time_data.shape[1] != 1):
            print(f"test_data dims  = {test_data.ndim}")
            print(f"test_data shape = {test_data.shape}")
            print(f"time_data dims  = {time_data.ndim}")
            print(f"time_data shape = {time_data.shape}")
            print("エラー：：次元数または形が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        if   self.regression == "n":  # 定数項なし&トレンドなし
            y_pred = np.dot(test_data, self.alpha)
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_1st * time_data
            
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_1st * time_data + self.trend_2nd * (time_data ** 2)
        
        return y_pred
    
    def dfRuller(self, qlist=[0.2, 0.8, 2.5, 5, 10]):
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        save_ρ = pd.read_csv("./csv_data/DF_distribution.csv.zst", header=0, compression="zstd").to_dict(orient="list")        
        
        if   self.regression == "n":  # 定数項なし&トレンドなし
            esti_coef = save_ρ["n"]
        elif self.regression == "c":  # 定数項あり&トレンドなし
            esti_coef = save_ρ["c"]
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            esti_coef = save_ρ["ct"]
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            # trend=ctt の場合のみ、サンプルデータの分布収束にバラツキが見られたため少し多めにデータを用意した。
            save_ρ2 = pd.read_csv("./csv_data/DF_distribution_ctt.csv.zst", header=0, compression="zstd").to_dict(orient="list")
            esti_coef = save_ρ["ctt"] + save_ρ2["ctt"]
        
        min_val = np.min(esti_coef)
        max_val = np.max(esti_coef)
        q_val   = np.percentile(esti_coef, q=qlist)
        
        dict_q = {}
        for q, val in zip(qlist, q_val):
            dict_q[str(q) + "%"] = val
        
        self.ρvalue = self.alpha[0, 0]
        if   self.ρvalue < min_val:
            self.pvalue = 0.0
        elif self.ρvalue > max_val:
            self.pvalue = 1.0
        else:
            np_coef = np.sort(esti_coef)
            num = len(esti_coef)
            index = np.abs(np_coef - self.ρvalue).argsort()[0].tolist()
            
            self.pvalue = (index + 1) / num
        
        return self.ρvalue, self.pvalue, self.lags, self.data_num, dict_q


class Augmented_Dickey_Fuller_Test:
    def __init__(self, test_data, regression="c") -> None:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if (test_data.ndim != 2) or (test_data.shape[1] != 1):
            print(f"test_data dims  = {test_data.ndim}")
            print(f"test_data shape = {test_data.shape}")
            print("エラー：：次元数または形が一致しません。")
            raise
        
        self.test_data           = test_data
        self.regression          = regression.lower()
        self.lags                = 0
        self.conv_data           = np.zeros([1, 1])
        self.train_data          = np.zeros([1, 1])
        self.alpha               = np.zeros([1, 1])
        self.alpha0              = np.zeros([1, 1])
        self.trend_1st           = np.zeros([1, 1])
        self.trend_2nd           = np.zeros([1, 1])
        self.sigma               = np.zeros([1, 1])
        self.data_num            = 0
        self.unbiased_dispersion = 0
        self.dispersion          = 0
        self.tvalue              = 0
        self.pvalue              = 0
        self.learn_flg           = False
    
    def fit(self, lags=1, offset=0) -> bool:
        # caution!!!
        # OLS(Ordinary Learst Squares)推定量を計算する際に
        # 擬似逆行列(pinv関数)を使用している箇所が存在する
        # この処理は逆行列が存在しない場合(行列式が0の場合)に発火する
        # しかし理論的には逆行列が存在しない時系列データの組み合わせは極めて稀である(無視できる)
        # 入力された時系列データ自体にミスが存在する(0の定数列になっている等)可能性が高い
        # statsmodels.tsa.vector_ar.var_model.VARではこのような時系列データを入力として与えた場合には
        # エラーを出力するようになっている
        # 本ライブラリにおいてエラーの出力を行わないのは、近似的にでも処理結果が欲しいためである
        
        tmp_train_data = np.diff(self.test_data, axis=0)
        tmp_train_data = tmp_train_data[offset:]
        nobs           = len(tmp_train_data)
        
        if not nobs - lags - tmp_train_data.shape[1] * lags - 1 > 0:
            # データ数に対して、最尤推定対象が多すぎる
            self.learn_flg = False
            return self.learn_flg
        
        x_data         = np.array([tmp_train_data[t-lags : t][::-1].ravel() for t in range(lags, nobs)])
        x_data         = np.hstack([self.test_data[offset+lags:-1, 0].reshape([x_data.shape[0], 1]), x_data])
        y_data         = self.test_data[offset+lags+1:]
        #y_data         = tmp_train_data[-len(x_data):]
        
        self.lags = lags
        num, s    = x_data.shape
        if   self.regression == "n":  # 定数項なし&トレンドなし
            A = x_data
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], np.zeros([1, x.shape[1]])
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            A = np.hstack([x_data, np.ones([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = np.zeros([1, x.shape[1]]), np.zeros([1, x.shape[1]])
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(1, num+1).reshape([num, 1])])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = x[s+1, :], np.zeros([1, x.shape[1]])
            
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            A = np.hstack([x_data, np.ones([num, 1]), np.arange(1, num+1).reshape([num, 1]), np.arange(1, num+1).reshape([num, 1]) ** 2])
            b = y_data
            try:
                # x = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
                x = np.linalg.solve(np.dot(A.T, A), np.dot(A.T, b))
            except np.linalg.LinAlgError as e:
                x = np.dot(np.linalg.pinv(np.dot(A.T, A)), np.dot(A.T, b))
            
            self.alpha,     self.alpha0    = x[0:s, :], x[s, :]
            self.trend_1st, self.trend_2nd = x[s+1, :], x[s+2, :]
            
        else:
            raise
        
        self.conv_data  = x_data
        self.train_data = A
        self.alpha      = self.alpha.reshape([s, x.shape[1]])
        self.alpha0     = self.alpha0.reshape([1, x.shape[1]])
        self.trend_1st  = self.trend_1st.reshape([1, x.shape[1]])
        self.trend_2nd  = self.trend_2nd.reshape([1, x.shape[1]])
        
        # なぜか、共分散行列の計算に特殊な計算方法が採用されている
        # statsmodels.tsa.vector_ar.var_model.VAR を参照のこと
        # _estimate_var関数内にて当該の記述を発見
        # どうやら不偏共分散の推定量らしい
        # math:: \frac{1}{T - Kp - 1} Y^\prime (I_T - Z (Z^\prime Z)^{-1} Z^\prime) Y
        # この式が元になっているらしい
        # よくわからないが、この式を採用することにする
        denominator   = y_data.shape[0] - y_data.shape[1] * lags - 1
        
        self.learn_flg = True
        y_pred         = self.predict(x_data, np.arange(1, num+1).reshape([num, 1]), isXDformat=True)
        diff           = y_pred - y_data
        self.sigma     = np.dot(diff.T, diff) / denominator
        self.data_num  = num
        self.unbiased_dispersion = denominator
        self.dispersion          = y_data.shape[0]

        return True
    
    def predict(self, test_data, time_data=np.array([[]]), isXDformat=False) -> np.ndarray:
        if type(test_data) is pd.core.frame.DataFrame:
            test_data = test_data.to_numpy()
        
        if type(time_data) is pd.core.frame.DataFrame:
            time_data = time_data.to_numpy()
        
        if type(test_data) is list:
            test_data = np.array(test_data)
        
        if type(time_data) is list:
            time_data = np.array(time_data)
        
        if (test_data.ndim != 2) or (time_data.ndim != 2) or (time_data.shape[1] != 1):
            print(f"test_data dims  = {test_data.ndim}")
            print(f"test_data shape = {test_data.shape}")
            print(f"time_data dims  = {time_data.ndim}")
            print(f"time_data shape = {time_data.shape}")
            print("エラー：：次元数または形が一致しません。")
            raise
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        if isXDformat == False:
            tmp_train_data = np.diff(test_data, axis=0)
            nobs           = len(tmp_train_data)
            x_data         = np.array([tmp_train_data[t-self.lags : t][::-1].ravel() for t in range(self.lags, nobs)])
            x_data         = np.hstack([self.test_data[self.lags:-1, 0].reshape([x_data.shape[0], 1]), x_data])
            
            test_data      = x_data
            time_data      = time_data[:len(test_data)]
        
        y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        if   self.regression == "n":  # 定数項なし&トレンドなし
            y_pred = np.dot(test_data, self.alpha)
            
        elif self.regression == "c":  # 定数項あり&トレンドなし
            y_pred = np.dot(test_data, self.alpha) + self.alpha0
        
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_1st * time_data
            
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            y_pred = np.dot(test_data, self.alpha) + self.alpha0 + self.trend_1st * time_data + self.trend_2nd * (time_data ** 2)
        
        return y_pred
    
    def log_likelihood(self) -> np.float64:
        # なぜか、対数尤度の計算に特殊な計算方法が採用されている
        # statsmodels.tsa.vector_ar.var_model を参照のこと
        # var_loglike関数内にて当該の記述を発見
        # どうやらこれで対数尤度を計算できるらしい
        # math:: -\left(\frac{T}{2}\right) \left(\ln\left|\Omega\right| - K\ln\left(2\pi\right) - K\right)
        # この式が元になっているらしい
        # さっぱり理解できないため、通常通りに計算することにする
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise

        y_data = self.test_data[self.lags:]
        num, _ = y_data.shape
        y_pred = self.predict(self.test_data, np.arange(1, num+1).reshape([num, 1]))

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
        
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        num = self.data_num
        k   = self.alpha.size + self.alpha0.size
        #log_likelihood = self.log_likelihood()
        
        # caution!!!
        # 本ライブラリでは、データ数に対して最尤推定対象が多い場合にもできる限り処理を続けるように調整してある
        # しかし、この場合に分散共分散行列の正定値性が保てなくなるという問題が発生する
        # また、入力された時系列データ自体に誤りが存在する場合にも正定値性が保てなくなる
        # 正定値行列でない場合には対数尤度の計算ができなくなる
        # この問題の対策のために対数尤度の近似値を求める処理に変更していることに注意
        # 参考URL
        # https://seetheworld1992.hatenablog.com/entry/2017/03/22/194932
        
        # 不偏推定共分散量を通常の推定共分散量に直す
        tmp_sigma = self.sigma * self.unbiased_dispersion / self.dispersion
        det_sigma = np.linalg.det(tmp_sigma)
        det_sigma = det_sigma if det_sigma != 0 else 1e-16

        inf = 0
        if ic == "aic":
            #inf = -2 * log_likelihood + 2 * k
            inf = np.log(np.abs(det_sigma)) + 2 * k / num
        elif ic == "bic":
            #inf = -2 * log_likelihood + k * np.log(num)
            inf = np.log(np.abs(det_sigma)) + k * np.log(num) / num
        elif ic == "hqic":
            #inf = -2 * log_likelihood + 2 * k * np.log(np.log(num))
            inf = np.log(np.abs(det_sigma)) + 2 * k * np.log(np.log(num)) / num
        else:
            raise

        return inf
    
    def select_order(self, maxlag=15, ic="aic", isVisible=False) -> int:
        if isVisible == True:
            print(f"AR model | {ic}", flush=True)

        nobs = len(self.test_data)
        if nobs <= maxlag:
            maxlag = nobs - 1

        model_param = []
        for lag in range(1, maxlag + 1):
            flg = self.fit(lags=lag, offset=maxlag - lag)
            
            if flg:
                rel = self.model_reliability(ic=ic)
                model_param.append([rel, lag])
            else:
                rel = np.finfo(np.float64).max
                model_param.append([rel, lag])

            if isVisible == True:
                print(f"AR({lag}) | {rel}", flush=True)
        
        res_rel, res_lag = np.finfo(np.float64).max, 0
        for elem in model_param:
            tmp_rel, tmp_lag = elem
            if res_rel > tmp_rel:
                res_rel = tmp_rel
                res_lag = tmp_lag
        
        res_lag = res_lag if res_lag != 0 else 1
        self.fit(lags=res_lag, offset=0)
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了しませんでした。")
            raise
        
        if isVisible == True:
            print(f"selected orders | {res_lag}", flush=True)

        return self.lags
    
    def adfRuller(self, qlist=[0.2, 0.8, 2.5, 5, 10]):
        if not self.learn_flg:
            print(f"learn_flg = {self.learn_flg}")
            print("エラー：：学習が完了していません。")
            raise
        
        save_ρ = pd.read_csv("./csv_data/ADF_distribution.csv.zst", header=0, compression="zstd").to_dict(orient="list")        
        
        if   self.regression == "n":  # 定数項なし&トレンドなし
            esti_coef = save_ρ["n"]
        elif self.regression == "c":  # 定数項あり&トレンドなし
            esti_coef = save_ρ["c"]
        elif self.regression == "ct": # 定数項あり&1次のトレンドあり
            esti_coef = save_ρ["ct"]
        elif self.regression == "ctt":# 定数項あり&1次のトレンドあり&2次のトレンドあり
            esti_coef = save_ρ["ctt"]
        
        min_val = np.min(esti_coef)
        max_val = np.max(esti_coef)
        q_val   = np.percentile(esti_coef, q=qlist)
        
        dict_q = {}
        for q, val in zip(qlist, q_val):
            dict_q[str(q) + "%"] = val
        
        self.ρvalue = self.alpha[0, 0]
        if   self.ρvalue < min_val:
            self.pvalue = 0.0
        elif self.ρvalue > max_val:
            self.pvalue = 1.0
        else:
            np_coef = np.sort(esti_coef)
            num = len(esti_coef)
            index = np.abs(np_coef - self.ρvalue).argsort()[0].tolist()
            
            self.pvalue = (index + 1) / num
        
        return self.ρvalue, self.pvalue, self.lags, self.data_num, dict_q

