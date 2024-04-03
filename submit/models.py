from django.db import models
class TrainParameters(models.Model):
    impute_model = models.TextField()
    predict_model = models.TextField()
    train_data_size = models.FloatField()
    predict_window_size = models.FloatField()
    imputation_size = models.FloatField()
    dataset = models.FileField(verbose_name='dataset',max_length=128,upload_to='dataset/')
# 存储训练结果

class TrainResult(models.Model):
    model = models.CharField(max_length=20)
    dataset = models.CharField(max_length=64)
    time = models.TimeField()
    accuracy = models.FloatField()
    precision = models.FloatField()
    SMAPE = models.FloatField()

class Task(models.Model):
    impute_model = models.CharField(max_length=20)
    predict_model = models.CharField(max_length=20)
    predict_window_size = models.FloatField()

class PredictResult(models.Model):
    index = models.IntegerField()
    true_value = models.FloatField()
    predict_value = models.FloatField()
    is_Anomaly = models.BooleanField()

class ImputeResult(models.Model):
    time = models.CharField(max_length=24)
    variable = models.IntegerField()
    Imputed_value = models.FloatField()

class AnomalyResult(models.Model):
    time = models.CharField(max_length=24)
    variable = models.IntegerField()
    true_value = models.FloatField()
    predict_value = models.FloatField()
    analysis = models.CharField(max_length=255, default='')
