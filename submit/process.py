import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import math
import os
import json
import heapq

import torch
import torch.nn as nn

from filterpy.kalman import KalmanFilter
from sklearn.preprocessing import StandardScaler

from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.mx.model.transformer import TransformerEstimator
from gluonts.mx.model.deepar import DeepAREstimator
from gluonts.mx.model.deep_factor import DeepFactorEstimator
from gluonts.mx.model.deepstate import DeepStateEstimator
from gluonts.mx.model.gp_forecaster import GaussianProcessEstimator
from gluonts.mx.model.n_beats import NBEATSEstimator
from gluonts.mx.model.simple_feedforward import SimpleFeedForwardEstimator

# from gluonts.nursery.SCott.model.lstm.lstm_estimator import LSTMEstimator
from gluonts.ext.prophet import ProphetPredictor

from gluonts.model.npts import NPTSPredictor
from gluonts.model.predictor import Predictor

from scaler import *

from pmdarima import auto_arima

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

import statsmodels.api as sm

from pathlib import Path

seq_len = 40
total_len = 143




def model_train(model_name, data, prediction_len):
    '''
    训练深度学习模型
    :param model_name: 需要训练的深度学习模型名，字符串格式
    :param data: 用于模型训练的数据，dataframe格式
    :param prediction_len: 预测长度
    :return: 训练好的模型参数所在文件路径(用于后续预测时加载模型)
    '''
    folder_path = "models/"
    save_path = ""
    predictor = None
    trainer = Trainer(epochs=5, learning_rate=1e-3, num_batches_per_epoch=100)
    if model_name == "LSTM":
        x_tensor, y_tensor = data_tensor(data, prediction_len)
        model = LSTMModel(output_size=prediction_len)
        loss_function = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        epochs = 50
        model.train()
        for e in range(epochs):
            out = model(x_tensor)
            loss = loss_function(out, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        file_path = 'models/LSTM.pth'
        torch.save(model.state_dict(), file_path)
        return file_path

    elif model_name == "Transformer":
        predictor = TransformerEstimator(
            freq="7D", inner_ff_dim_scale=4, num_heads=8, prediction_length=prediction_len, trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'Transformer/'


    elif model_name == "DeepAR":
        predictor = DeepAREstimator(
            freq="7D", num_layers=2, num_cells=40, prediction_length=prediction_len, trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'DeepAR/'

    elif model_name == "DeepFactor":
        predictor = DeepFactorEstimator(
            prediction_length=prediction_len, freq="7D", cell_type="lstm", num_factors=10, trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'DeepFactor/'

    elif model_name == "DeepState":
        predictor = DeepStateEstimator(
            freq="7D", prediction_length=prediction_len, num_layers=2, num_cells=40,
            use_feat_static_cat=False, cardinality=[1], trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'DeepState/'

    elif model_name == "GPForecaster":
        predictor = GaussianProcessEstimator(
            freq="7D", prediction_length=prediction_len, max_iter_jitter=10, cardinality=data.shape[0], trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'GPForecaster/'

    elif model_name == "NBeats":
        predictor = NBEATSEstimator(
            freq="7D", prediction_length=prediction_len, loss_function="MAPE", trainer=trainer
        ).train(data_list_dataset(data))
        save_path = 'NBeats/'

    else:
        raise Exception('模型名不存在！')

    path = folder_path + save_path

    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)

    predictor.serialize(Path(path))

    return path


def model_predict(model_name, path, is_dl, prediction_len=10, history_data=None):
    '''
    :param path: 路径名
    :param history_data: 历史数据集，dataframe格式
    :param prediction_len: 预测长度
    :param model_name: 模型名
    :param is_dl: True深度学习模型，False机器学习模型
    :return: 预测结果
    '''
    pred_res = []
    history_list_dataset = None
    try:
        history_list_dataset = data_list_dataset(history_data)
    except Exception:
        raise Exception
    else:
        if is_dl:
            if model_name == "LSTM":
                hs_tensor = data_tensor(history_data, 0)
                model = LSTMModel(output_size=prediction_len)
                model.load_state_dict(torch.load(path))
                pred_res = model(hs_tensor)
                return pred_res.detach().numpy().squeeze()

            else:
                predictor = Predictor.deserialize(Path(path))
                forecast_it = predictor.predict(history_list_dataset, num_samples=100)
                fore = list(forecast_it)
                pred_res = np.mean(fore[0].samples, axis=0)

        else:
            if model_name == "ETS":
                predictor = ETSModel(history_data.squeeze(), error="add", trend="additive", seasonal="add",
                                     seasonal_periods=4).fit()
                pred_res = predictor.forecast(prediction_len)

            elif model_name == "Prophet":
                predictor = ProphetPredictor(
                    prediction_length=prediction_len,
                    prophet_params={'changepoint_prior_scale': 0.03, 'changepoint_range': 0.5}
                )
                forecast_it = predictor.predict(history_list_dataset, num_samples=100)
                pred_res = np.mean(list(forecast_it)[0].samples, axis=0)

            elif model_name == "NPTS":
                predictor = NPTSPredictor(prediction_length=prediction_len)
                forecast_it = predictor.predict(history_list_dataset, num_samples=100)
                pred_res = np.mean(list(forecast_it)[0].samples, axis=0)

            elif model_name == "ARIMA":
                model_search = auto_arima(history_data.squeeze(), information_criterion='aic', seasonal_test='ocsb')
                predictor = ARIMA(history_data.squeeze(), order=model_search.order).fit()
                pred_res = predictor.forecast(prediction_len)

            elif model_name == "Holt-Winters":
                predictor = ExponentialSmoothing(history_data.squeeze(),
                                                 seasonal_periods=4, seasonal='additive').fit()
                pred_res = predictor.forecast(prediction_len)

            elif model_name == "Linear":
                # estimator = SimpleFeedForwardEstimator(
                #     prediction_length=43,  # 预测长度
                #     context_length=10,  # 上下文长度
                #     trainer=Trainer(epochs=5, learning_rate=1e-3, num_batches_per_epoch=100)
                # )
                # predictor = estimator.train(history_list_dataset)
                # forecast_it = predictor.predict(history_list_dataset)
                # fore = list(forecast_it)
                # pred_res = np.mean(fore[0].samples, axis=0)

                # 拟合Holt's Linear Trend模型
                predictor = sm.tsa.Holt(history_data.squeeze(), damped=True).fit()  # 使用damped=True引入阻尼趋势
                pred_res = predictor.forecast(steps=prediction_len)

            elif model_name == "Period":
                # 训练SARIMA模型
                # 季节性周期长度为52（一年的数据点数量）
                # 季节性阶数和趋势阶数根据数据的特性选择
                model = SARIMAX(history_data.squeeze(), order=(1, 1, 1), seasonal_order=(1, 1, 1, 52)).fit()
                pred_res = model.forecast(steps=prediction_len)

            else:
                pass

        return pred_res

def model_selection(top_k=3, class_num=0,strategy="best_acc"):
    '''

    :param strategy: 筛选策略，默认accuracy优先
    :param top_k: 需要筛选出前k个模型
    :param class_num: 当前数据所属类
    :return: 前k个最佳模型
    '''
    with open("features-lag.json", 'r', encoding='utf-8') as file:
        # 加载JSON内容
        features = json.load(file)

    # 实际环境中，从数据库中读取出每个样本的最优模型
    with open("best-original.json", 'r', encoding='utf-8') as file:
        # 加载JSON内容
        best = json.load(file)

    res = []
    # 找出当前类别下所有样本对应索引
    idx = [i for i in range(len(features)) if features[i]['class'] == class_num]
    # 根据索引得到每个样本对应的最佳模型
    curr_best = [best[i] for i in idx]
    # 大根堆选出top-k，依据accuracy
    candidates = heapq.nlargest(top_k, curr_best, key=lambda x: x[strategy])
    # 去重
    option = []
    for i in candidates:
        if i['best_model'] not in option:
            option.append(i['best_model'])

    return option

def data_process(df=None,prediction_window=0.1):
    '''
    :param prediction_window: 预测窗口比例,(0,1)的浮点数
    :param data: 从源文件读取出的dataframe，此处实例假设不需要分组
    :return: 训练集dataframe格式，测试集数组格式，预测窗口长度
    '''
    df = pd.read_csv('Walmart.csv') if df is None else df
    # 处理日期格式
    df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
    df.set_index('Date', inplace=True)


    # 实际情况根据具体列分组
    grouped = df.groupby(df['Store'])

    # 卡尔曼滤波器，对数据进行平滑处理
    kf = KalmanFilter(dim_x=1, dim_z=1)
    kf.F = np.array([[1.]])
    kf.H = np.array([[1.]])
    kf.Q = 0.01
    kf.R = 0.1
    kf.x = np.array([[df['Weekly_Sales'].iloc[0]]])
    kf.P = np.eye(1)

    smoothed_weekly_sales = []
    for weekly_sales in df['Weekly_Sales']:
        kf.predict()
        kf.update(weekly_sales)
        smoothed_weekly_sales.append(kf.x[0, 0])

    # 将平滑后的数据添加到 DataFrame 中
    df['Smoothed_Weekly_Sales'] = smoothed_weekly_sales

    # 保留需要的列
    data_list = []
    columns_to_drop = [col for col in df.columns if col not in ['Smoothed_Weekly_Sales']]
    for index, group in grouped:
        group = group.drop(columns=columns_to_drop)
        data_list.append(group)

    # 只取第一组数据
    data = data_list[0]
    total_len = data.shape[0]
    window_len = math.floor(prediction_window * total_len)
    hs_df = data[:(total_len - window_len)]
    test_df = data[(total_len - window_len):]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(hs_df['Smoothed_Weekly_Sales'].values.reshape(-1,1))

    hs_df = pd.DataFrame(X_scaled, index=hs_df.index, columns=['Smoothed_Weekly_Sales'])
    test_l = scaler.transform(test_df).squeeze()

    return hs_df,test_l,window_len

# 把数据处理成张量
def data_tensor(df, window_len):
    if window_len == 0:
        df_array = np.array(df)
        df_tensor = torch.tensor(df_array.reshape(1, df_array.shape[0], 1), dtype=torch.float32)
        return df_tensor
    else:
        x_array = np.array(df[:-window_len])
        y_array = np.array(df[-window_len:])
        x = torch.tensor(x_array.reshape(1, x_array.shape[0], 1), dtype=torch.float32)
        y = torch.tensor(y_array.reshape(1, y_array.shape[0]), dtype=torch.float32)
        return x, y


def data_list_dataset(data):
    '''
    :param data: dataframe格式
    :return: gluonts模型所需的输入数据格式
    '''
    train_data = None
    try:
        train_data = ListDataset(
            [{"start": data.index[0], "target": data.squeeze()}],
            freq="7D"
        )
    except Exception:
        raise Exception("数据为空")

    return train_data

# 构建LSTM模型
class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=40, num_layers=2, output_size=1):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out


# 保存数据到文件
def save_file(res, file_name):
    '''
    :param res: 结果列表
    :param file_name: 文件名
    '''
    # 检查文件是否存在并且是否为空
    if os.path.exists(file_name) and os.stat(file_name).st_size != 0:
        # 文件不为空，读取文件中的内容并解析为列表
        with open(file_name, "r") as file:
            models_list = json.load(file)
    else:
        # 文件为空，创建空列表
        models_list = []

    # 将当前对象加入到列表中
    models_list.append(res)

    # 将列表写入JSON文件
    with open(file_name, "w") as file:
        json.dump(models_list, file)
