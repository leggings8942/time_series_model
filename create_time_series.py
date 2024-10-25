import numpy as np

def create_trend_data(x, regression="n"):
    length = len(x)
    const  = np.ones([length, 1])
    trend  = np.arange(1, length+1).reshape([length, 1])
    if   regression == "n":
        trend_data = x
    elif regression == "c":
        trend_data = np.hstack([x, const])
    elif regression == "ct":
        trend_data = np.hstack([x, const, trend])
    elif regression == "ctt":
        trend_data = np.hstack([x, const, trend, trend ** 2])
    else:
        raise
    
    return trend_data

def create_time_data_n(length=1000, scale=1):
    x = np.random.normal(loc=0, scale=scale, size=[length])
    return x.cumsum()

def create_time_data_n_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1):
    x = np.empty(length)
    x[0] = 0
    for idx in range(1, lags):
        tmp = [coef[-idx2] * x[idx - 1 - idx2] for idx2 in range(0, idx)]
        x[idx] = np.sum(tmp) + np.random.normal(loc=0, scale=scale)
    
    for idx in range(lags, length):
        x[idx] = np.sum(coef * x[idx-lags:idx]) + np.random.normal(loc=0, scale=scale)
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = create_time_data_n_lag(lags, length, scale) 
    
    return x

def create_time_data_c(length=1000, scale=1, const=10):
    x = np.random.normal(loc=0, scale=scale, size=[length])
    return const + x.cumsum()

def create_time_data_c_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10):
    x = np.empty(length)
    x[0] = const
    for idx in range(1, lags):
        tmp = [coef[-idx2] * x[idx - 1 - idx2] for idx2 in range(0, idx)]
        x[idx] = np.sum(tmp) + np.random.normal(loc=0, scale=scale)
    
    for idx in range(lags, length):
        x[idx] = np.sum(coef * x[idx-lags:idx]) + np.random.normal(loc=0, scale=scale)
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = create_time_data_c_lag(lags, length, scale, const)
    
    return x

def create_time_data_ct(length=1000, scale=1, const=10, δ1=2):
    x = δ1 + np.random.normal(loc=0, scale=scale, size=[length])
    return const + x.cumsum()

def create_time_data_ct_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10, δ1=2):
    x = np.empty(length)
    x[0] = const
    for idx in range(1, lags):
        tmp = [coef[-idx2] * x[idx - 1 - idx2] for idx2 in range(0, idx)]
        x[idx] = np.sum(tmp) + δ1 + np.random.normal(loc=0, scale=scale)
    
    for idx in range(lags, length):
        x[idx] = δ1 + np.sum(coef * x[idx-lags:idx]) + np.random.normal(loc=0, scale=scale)
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = create_time_data_ct_lag(lags, length, scale, const, δ1)
    
    return x

def create_time_data_ctt(length=1000, scale=1, const=10, δ1=2, δ2=1.1):
    t = np.arange(1, length + 1)
    x = δ1 + δ2 * t + np.random.normal(loc=0, scale=scale, size=[length])
    return const + x.cumsum()

def create_time_data_ctt_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10, δ1=2, δ2=1.1):
    x = np.empty(length)
    x[0] = const
    for idx in range(1, lags):
        tmp = [coef[-idx2] * x[idx - 1 - idx2] for idx2 in range(0, idx)]
        x[idx] = np.sum(tmp) + δ1 + (idx + 1) * δ2 + np.random.normal(loc=0, scale=scale)
    
    for idx in range(lags, length):
        x[idx] = δ1 + (idx + 1) * δ2 + np.sum(coef * x[idx-lags:idx]) + np.random.normal(loc=0, scale=scale)
    
    if np.any(np.isnan(x)) or np.any(np.isinf(x)):
        x = create_time_data_ctt_lag(lags, length, scale, const, δ1, δ2)
    
    return x

def create_sample_data_n(length=1000, scale=1):
    x = create_time_data_n(length, scale)
    
    train_data = x.reshape([length, 1])
    x_data     = train_data[:length-1]
    y_data     = train_data[1:]
    
    A     = create_trend_data(x_data, regression="n")
    b     = y_data
    alpha = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_n_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1):
    x = create_time_data_n_lag(lags, length, coef, scale)
    
    train_data = np.diff(x.reshape([length, 1]), axis=0)
    x_data     = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, length-1)])
    x_data     = np.hstack([x[lags:-1].reshape([x_data.shape[0], 1]), x_data])
    y_data     = x[lags+1:].reshape([x_data.shape[0], 1])
    
    A     = create_trend_data(x_data, regression="n")
    b     = y_data
    alpha = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_c(length=1000, scale=1, const=10):
    x = create_time_data_c(length, scale, const)
    
    train_data = x.reshape([length, 1])
    x_data     = train_data[:length-1]
    y_data     = train_data[1:]
    
    A     = create_trend_data(x_data, regression="c")
    b     = y_data
    alpha = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_c_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10):
    x = create_time_data_c_lag(lags, length, coef, scale, const)
    
    train_data = np.diff(x.reshape([length, 1]), axis=0)
    x_data     = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, length-1)])
    x_data     = np.hstack([x[lags:-1].reshape([x_data.shape[0], 1]), x_data])
    y_data     = x[lags+1:].reshape([x_data.shape[0], 1])
    
    A     = create_trend_data(x_data, regression="c")
    b     = y_data
    alpha = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_ct(length=1000, scale=1, const=10, δ1=2):
    x = create_time_data_ct(length, scale, const, δ1)
    
    train_data = x.reshape([length, 1])
    x_data     = train_data[:length-1]
    y_data     = train_data[1:]
    
    A     = create_trend_data(x_data, regression="ct")
    b     = y_data
    alpha = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_ct_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10, δ1=2):
    x = create_time_data_ct_lag(lags, length, coef, scale, const, δ1)
    
    train_data = np.diff(x.reshape([length, 1]), axis=0)
    x_data     = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, length-1)])
    x_data     = np.hstack([x[lags:-1].reshape([x_data.shape[0], 1]), x_data])
    y_data     = x[lags+1:].reshape([x_data.shape[0], 1])
    
    A     = create_trend_data(x_data, regression="ct")
    b     = y_data
    alpha = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_ctt(length=1000, scale=1, const=10, δ1=2, δ2=1.1):
    x = create_time_data_ctt(length, scale, const, δ1, δ2)
    
    train_data = x.reshape([length, 1])
    x_data     = train_data[:length-1]
    y_data     = train_data[1:]
    
    A     = create_trend_data(x_data, regression="ctt")
    b     = y_data
    alpha = np.dot(np.linalg.inv(np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_sample_data_ctt_lag(lags=2, length=1000, coef=np.array([0.5, 0.5]), scale=1, const=10, δ1=2, δ2=1.1):
    x = create_time_data_ctt_lag(lags, length, coef, scale, const, δ1, δ2)
    
    train_data = np.diff(x.reshape([length, 1]), axis=0)
    x_data     = np.array([train_data[t-lags : t][::-1].ravel() for t in range(lags, length-1)])
    x_data     = np.hstack([x[lags:-1].reshape([x_data.shape[0], 1]), x_data])
    y_data     = x[lags+1:].reshape([x_data.shape[0], 1])
    
    A     = create_trend_data(x_data, regression="ctt")
    b     = y_data
    alpha = np.dot(np.linalg.inv( np.dot(A.T, A)), np.dot(A.T, b))
    
    return x[-1], alpha[0, 0]

def create_time_data(input_list):
    regression, length, scale, const, δ1, δ2 = input_list
    
    if   regression == "n":
        x, alpha = create_sample_data_n(  length, scale)
    elif regression == "c":
        x, alpha = create_sample_data_c(  length, scale, const)
    elif regression == "ct":
        x, alpha = create_sample_data_ct( length, scale, const, δ1)
    elif regression == "ctt":
        x, alpha = create_sample_data_ctt(length, scale, const, δ1, δ2)
    else:
        raise
    
    return x, alpha

def create_time_data_lag(input_list):
    regression, lags, length, coef, scale, const, δ1, δ2 = input_list
    
    if   regression == "n":
        x, alpha = create_sample_data_n_lag(  lags, length, coef, scale)
    elif regression == "c":
        x, alpha = create_sample_data_c_lag(  lags, length, coef, scale, const)
    elif regression == "ct":
        x, alpha = create_sample_data_ct_lag( lags, length, coef, scale, const, δ1)
    elif regression == "ctt":
        x, alpha = create_sample_data_ctt_lag(lags, length, coef, scale, const, δ1, δ2)
    else:
        raise
    
    return x, alpha