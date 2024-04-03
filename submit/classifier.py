import numpy as np
import pandas as pd
import json

from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from keras.models import Sequential
from keras.layers import Dense
from keras.utils.np_utils import to_categorical

import xgboost as xgb
import lightgbm as lgb

def clustering(n_clusters,data=None):
    '''
    :param n_clusters: 聚类中心数
    :param data: 待聚类数据，dataframe格式
    :return: 聚类后标签
    '''

    features = pd.read_json('features.json') if data is None else data

    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(features)
    labels = kmeans.labels_

    return labels

    # res = []
    # for i,item in enumerate(json.loads(features.to_json(orient='records'))):
    #     item['class'] = int(labels[i])
    #     res.append(item)
    # # 获取聚类标签
    # with open('features-lag.json', 'w') as file:
    #     json.dump(res, file)



def classifier(classifier_name,train_x,train_labels,n_clusters,test_x):
    '''
    :param classifier_name: 所选分类器名
    :param train_x: 训练集特征
    :param train_labels: 训练集分类标签
    :param n_clusters: 聚类中心数
    :param test_x: 测试集特征
    :return: 测试集分类结果
    '''
    test_labels = None
    if classifier_name == "SVM":
        # 实例化 SVM 分类器对象并拟合
        classifier = SVC(kernel='linear', C=1.0, random_state=42).fit(train_x, train_labels)
        test_labels = classifier.predict(test_x)

    elif classifier_name == "XGBoost":
        # 转换数据为 DMatrix 格式
        dtrain = xgb.DMatrix(train_x, label=train_labels)
        # 设置参数
        params = {
            'objective': 'multi:softmax',  # 多分类问题
            'num_class': n_clusters,  # 类别数
            'eval_metric': 'merror'  # 评估指标为分类错误率
        }
        # 训练模型
        num_round = 100
        classifier = xgb.train(params, dtrain, num_round)
        test_labels = classifier.predict(test_x)

    elif classifier_name == "RandomForest":
        classifier = RandomForestClassifier(n_estimators=100, random_state=42).fit(train_x, train_labels)
        test_labels = classifier.predict(test_x)

    elif classifier_name == "LightGbm":
        train_data = lgb.Dataset(train_x, label=train_labels)
        # 设置参数
        params = {
            'objective': 'multiclass',  # 多分类问题
            'num_class': n_clusters,  # 类别数
            'metric': 'multi_error'  # 评估指标，多分类错误率
        }
        num_round = 100  # 迭代轮数
        bst = lgb.train(params, train_data, num_round)
        # 预测测试集的类别标签
        test_pred = bst.predict(test_x, num_iteration=bst.best_iteration)
        test_labels = np.argmax(test_pred, axis=1)  # 获取概率最大的类别索引

    elif classifier_name == "KNN":
        # 实例化KNN分类器
        classifier = KNeighborsClassifier(n_neighbors=n_clusters).fit(train_x, train_labels)
        test_labels = classifier.predict(test_x)

    elif classifier_name == "CNN":
        model = Sequential()
        model.add(Dense(units=64, activation='relu', input_shape=(train_x.shape[1],)))
        model.add(Dense(units=32, activation='relu'))
        model.add(Dense(units=n_clusters, activation='softmax'))  # 输出层，n个单元对应n个类别
        # 编译模型
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        # 训练模型
        model.fit(train_x, to_categorical(train_labels, num_classes=n_clusters), epochs=100,
                                                                    batch_size=5, verbose=1, )
        predictions = model.predict(test_x)
        test_labels = np.argmax(predictions, axis=1)

    else:
        raise Exception("分类器不存在")

    return test_labels


