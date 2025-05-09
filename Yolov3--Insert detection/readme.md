# 本实验是手搓YOlov3模型进行昆虫数据集检测(Pytorch版本)
## 模型训练
~~~
tensorboard --logdir=runs
~~~
~~~
python train.py
~~~
## 模型测试
~~~
python test.py
~~~

## 测试结果
![image](https://github.com/user-attachments/assets/0839e50b-2a5b-468f-9383-3eb8e2a25c91)
![image](https://github.com/user-attachments/assets/7b391500-9631-417c-883e-09e4cebcedd0)
![image](https://github.com/user-attachments/assets/94128712-75fb-42c1-a08c-f9221ff6a264)
![image](https://github.com/user-attachments/assets/fc47ca15-b742-4116-b0a7-835a0e95ceee)

## loss变化与mAP变化
训练loss:

![image](https://github.com/user-attachments/assets/e78ea926-e1db-4be0-a0de-93e8319f492b)

验证loss:

![image](https://github.com/user-attachments/assets/dcbc7902-58c7-4dc0-a47e-36fe5d6aed77)

训练/验证map:

![image](https://github.com/user-attachments/assets/20c90091-b0cd-4d1a-be4a-74a7fe15cdea)

