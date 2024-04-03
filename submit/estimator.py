import numpy as np
import statsmodels.api as sm

# accuracy = 1-MAPE
def cal_accuracy(predicted,actual):
    np_acc = 1-np.abs((predicted-actual)/actual)
    np_acc = [0 if ((x<0) or (np.isnan(x)) or (np.isinf(x))) else x for x in np_acc]
    return np.mean(np_acc)


# precision = TP/Total,tolerance初始为3%
def cal_precision(predicted,actual,tolerance=0.03):
    return sum((np.abs(predicted-actual)/np.abs(actual)) <= tolerance)/len(actual)


def cal_smape(predicted,actual):
    # n = len(predicted)
    # total_smape = 0
    # for i in range(n):
    #     # 计算每个数据点的 SMAPE
    #     numerator = abs(predicted[i] - actual[i])
    #     denominator = (abs(predicted[i]) + abs(actual[i])) / 2
    #     if denominator != 0:  # 避免分母为0的情况
    #         total_smape += (numerator / denominator)
    #
    # smape = (total_smape / n) * 100
    # return smape
    return 2.0 * np.mean(np.abs(predicted-actual)/(np.abs(predicted)+np.abs(actual)))

def cal_manual_features(ts_l):
    features = {}
    # 计算原数据的方差
    features['variance'] = np.var(ts_l)
    # 计算自相关函数来检查周间周期性
    features['periodicity'] = sm.tsa.acf(ts_l, nlags=1)
    # 计算最大值
    features['max'] = np.max(ts_l)
    # 计算最小值
    features['min'] = np.min(ts_l)
    # 计算均值
    features['mean'] = np.mean(ts_l)
    # 计算绝对方差
    features['ab_variance'] = np.mean(np.abs(ts_l - np.mean(ts_l)))

    return features



# 3.存储每个数据集对应的所有模型评估指标，后续聚类