### TENDS: A Time Series Management System based on Model Selection
### Please check the 'master' branch

This repository contains the frontend source code and portions of the backend code for TENDS, as featured in the paper "A Demonstration of TENDS: Time Series Management System based on Model Selection".

TENDS is a comprehensive time series management system designed to improve data quality and analysis in handling a diverse range of time series data. The interface of TENDS consists of two components, including offline training and online management. This system addresses the functionality and adaptability gaps in existing time series management systems.It's standout features include:
(1) An effective model selection mechanism tailored to improve efficacy across varied data types.
(2) Offers fourteen state-of-the-art predictive methods and three cutting-edge imputation methods.
(3) An evolving dynamic expert knowledge base for anomaly detection ensures ongoing accuracy with new data.

**offline training**  
![screenshot](https://github.com/IAA111/SimpleTSDemo/blob/main/media/offline.png)

**online management**  
![screenshot](https://github.com/IAA111/SimpleTSDemo/blob/main/media/online.png)

**(1) migrate the database**  
```
python manage.py makemigrations    
python manage.py migrate
```
**(2) startup frontend**
```
python manage.py runserver
```

**(3) open the web page**  
[http://127.0.0.1:8000](http://127.0.0.1:8000)
