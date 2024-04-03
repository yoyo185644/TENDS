import asyncio
import json
import time
from asgiref.sync import sync_to_async
from submit.models import TrainParameters
from submit.models import Task
from channels.consumer import AsyncConsumer
import csv
import pandas as pd
from . import models
from . import views

from process import *
from estimator import *

class TrainChatConsumer(AsyncConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.training_task = None
        self.impute_status = None
        self.predict_status = None
        self.impute_start_time = None
        self.predict_start_time = None
        self.impute_total_model = None
        self.impute_model_count = None
        self.predict_total_model = None
        self.predict_model_count = None
        self.impute_model = None
        self.predict_model = None
        self.train_data_size = None
        self.predict_window_size = None
        self.imputation_size = None
        self.dataset = None

    # WebSocket连接成功
    async def websocket_connect(self, event):
        print("connected", event)
        await self.send({
            "type": "websocket.accept"
        })

    # 当前端发送消息到服务器时
    async def websocket_receive(self,event):
        print("received", event)
        text_data = json.loads(event['text'])
        message = text_data['type']

        if message == "training.start":
            if self.training_task:
                self.training_task.cancel()
            self.training_task = asyncio.ensure_future(self.start_training())

        elif message == "training.stop":
            if self.training_task:
                self.training_task.cancel()
                self.training_task = None

    # WebSocket连接断开
    async def websocket_disconnect(self, event):
        print("disconnected", event)

    async def start_training(self):
        # 获取训练参数
        model_parameters = await sync_to_async(TrainParameters.objects.last, thread_sensitive=True)()
        self.impute_model = model_parameters.impute_model.split(',')
        self.predict_model = model_parameters.predict_model.split(',')
        self.train_data_size = model_parameters.train_data_size
        self.predict_window_size = model_parameters.predict_window_size
        self.imputation_size = model_parameters.imputation_size
        self.impute_model_count = 0
        self.predict_model_count = 0
        self.impute_total_model = len(self.impute_model)
        self.predict_total_model = len(self.predict_model)

        if model_parameters.dataset:
            self.dataset = model_parameters.dataset.open('r')            # 读取 dataset 文件

        await self.impute()
        self.dataset.close()                                             # 关闭 dataset 文件
        await self.train_all_models()

    async def send_status(self):
        await self.send({
            "type": "websocket.send",
            "text": json.dumps({
                "impute_status": self.impute_status,
                "impute_start_time": self.impute_start_time,
                "predict_status": self.predict_status,
                "predict_start_time": self.predict_start_time,
                "impute_total_model": self.impute_total_model,
                "impute_model_count":self.impute_model_count,
                "predict_total_model": self.predict_total_model,
                "predict_model_count": self.predict_model_count,
            })
        })

    async def impute(self):
        self.impute_start_time = time.time()
        self.impute_status = "progressing"
        self.predict_status = "Not Started"
        await self.send_status()

        df = pd.read_csv(self.dataset)             # 读取所有数据
        print(df)

        await asyncio.sleep(10)

        '''
             补全过程 根据补全模型选择合适的补全方法进行补全
             将补全结果保存在数据库中
        '''

        self.impute_status = "finished"
        await self.send_status()
        print("impute complete")

    async def train_all_models(self):
        self.predict_start_time = time.time()
        self.predict_status = "Progressing"
        await self.send_status()


        ''' 
        从数据库中获取已经补全的数据
        先把 TrainResult 清空
        对每个预测模型进行训练 
            
        '''
        # 根据实际情况传入数据和预测窗口大小
        prediction_window = 0.1

        hs_df,_,_ = data_process(df=None,prediction_window=prediction_window)

        # predict_model_choice应为深度学习模型名列表
        for model in predict_model_choice:
            start = time.time

            # 得到模型路径,需要数据库存储path
            path = model_train(model, hs_df, prediction_window)

            time = time.time() - start

            # model_count += 1

            # 发送当前训练状态
            await self.send_status()

            # 将该模型训练结果保存到数据表中
            form = views.TrainResultForm()
            form.model = model
            # form.time = time
            # form.accuracy = accuracy
            # form.precision = precision
            # form.SMAPE = SMAPE

            if form.is_valid():
                form.save()
            else:
                print(form.errors)
        await asyncio.sleep(10)

        self.predict_status = "finished"
        await self.send_status()


class TaskChatConsumer(AsyncConsumer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.task = None
        self.start_time = None
        self.status = None
        self.impute_model = None
        self.predict_model = None
        self.predict_window_size = None

    async def websocket_connect(self, event):
        print("connected", event)
        await self.send({
            "type": "websocket.accept"
        })

    async def websocket_receive(self, event):
        print("received", event)
        text_data = json.loads(event['text'])
        message = text_data['type']

        if message == "task.start":
            if self.task:
                self.task.cancel()
            self.task = asyncio.ensure_future(self.start_task())

        elif message == "task.stop":
            if self.task:
                self.task.cancel()
                self.task = None

    async def websocket_disconnect(self, event):
        print("disconnected", event)

    async def start_task(self):
        # 获取参数
        task_parameters = await sync_to_async(Task.objects.last, thread_sensitive=True)()
        self.impute_model = task_parameters.impute_model
        self.predict_model = task_parameters.predict_model
        self.predict_window_size = task_parameters.predict_window_size

        self.start_time = time.time()
        self.status = "progressing"
        await self.send_status()

        await self.impute()
        await self.predict()

        self.status = "finished"
        await self.send_status()

    async def send_status(self):
        await self.send({
            "type": "websocket.send",
            "text": json.dumps({
                "status": self.status,
                "start_time": self.start_time,
            })
        })

    async def impute(self):
        nan_counts_dict = {"figure1": 2, "figure2": 150, "figure3": 3, "figure4": 6, "figure5": 8, "figure6": 10}
        await self.send({
            "type": "websocket.send",
            "text": json.dumps({
                "impute_data": nan_counts_dict
            })
        })

        print("开始执行补全")

        '''  

             具体补全过程

        '''
        await asyncio.sleep(10)


    async def predict(self):
        print("开始执行预测")

        '''
            记录最佳模型
            best_model = None
            best_acc = 0
            best_precision = 0
            best_smape = float('inf')
        '''
        prediction_window = 0.1
        hs_df, test_l,window_len = data_process(df=None, prediction_window=prediction_window)
        window = self.predict_window_size
        # dl_list:深度学习模型名字列表
        for dl in dl_list:
            print(dl)
            # 开始预测计时
            start = time.time()
            # path从数据库中取，
            pred_l = model_predict(model_name=dl, path=path, is_dl=True,prediction_len=window_len , history_data=None)
            # 预测结束
            interval = time.time() - start

            acc = cal_accuracy(pred_l, test_l)
            precision = cal_precision(pred_l, test_l)
            smape = cal_smape(pred_l, test_l)

            '''
                if acc>best_acc:
                    best_acc = acc
                    best_model = dl
                best_precision = precision if precision>best_precision else best_precision
                best_smape = smape if smape<best_smape else best_smape
            '''


        for ml in ml_list:
            print(ml)
            # 开始预测计时
            start = time.time()
            # path从数据库中取，
            pred_l = model_predict(model_name=ml, path=None, is_dl=False,prediction_len=window_len , history_data=None)
            # 预测结束
            interval = time.time() - start

            acc = cal_accuracy(pred_l, test_l)
            precision = cal_precision(pred_l, test_l)
            smape = cal_smape(pred_l, test_l)

            '''
                if acc>best_acc:
                    best_acc = acc
                    best_model = dl
                best_precision = precision if precision>best_precision else best_precision
                best_smape = smape if smape<best_smape else best_smape
            '''

        await asyncio.sleep(5)

