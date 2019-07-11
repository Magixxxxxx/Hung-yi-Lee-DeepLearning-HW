import sys
import numpy as np
import pandas as pd
import csv

raw_data = np.genfromtxt('train.csv',delimiter=',',encoding='ansi')#只处理数字？
data = raw_data[1:,3:]#表示取第一行以后，和第三列以后的数据
where_are_NaNs = np.isnan(data) #一个numpy的ndarray的对象，特点是不复制
data[where_are_NaNs] = 0 #ndarray的特殊操作
month_to_data = {}  # Dictionary (key:month , value:data)                                  

print(data)

for month in range(12):
    sample = np.empty(shape = (18 , 480))  #18个属性，20天*24小时，sample(x,y)代表属性x在y时的值
    for day in range(20): 
        for hour in range(24): 
            sample[:,day * 24 + hour] = data[18 * (month * 20 + day): 18 * (month * 20 + day + 1),hour]
    month_to_data[month] = sample  #每月前20天是训练集,12*18*480

x = np.empty(shape = (12 * 471 , 18 * 9),dtype = float)
y = np.empty(shape = (12 * 471 , 1),dtype = float)#每个月480-9=471个样本(每连续九小时取样，最后9小时无法取样)

for month in range(12): 
    for day in range(20): 
        for hour in range(24):   
            if day == 19 and hour > 14:
                continue  
            x[month * 471 + day * 24 + hour,:] = month_to_data[month][:,day * 24 + hour : day * 24 + hour + 9].reshape(1,-1) 
            #化矩阵为新的行列数，-1代表自动推断,这里去掉了18*9的二维属性，转而以一维序列代替，一维序列的顺序本身可以隐含其时序信息
            y[month * 471 + day * 24 + hour,0] = month_to_data[month][9 ,day * 24 + hour + 9]

mean = np.mean(x, axis = 0) #aix=0表示沿每列计算
std = np.std(x, axis = 0) #标准差
for i in range(x.shape[0]):
    for j in range(x.shape[1]):
        if not std[j] == 0 :
            x[i][j] = (x[i][j]- mean[j]) / std[j] #所有属性归一化

dim = x.shape[1] + 1 
w = np.zeros(shape = (dim, 1 )) #empty创建一个数组，数组中的数取决于数组在内存中的位置处的值，为0纯属巧合？
x = np.concatenate((np.ones((x.shape[0], 1 )), x) , axis = 1).astype(float)#给x的162个属性补1

#初始化学习率(163个参数，163个200)和adagrad
learning_rate = np.array([[200]] * dim)
adagrad_sum = np.zeros(shape = (dim, 1 ))

#没有隐藏层的网络
for T in range(10001):
    if(T % 500 == 0 ):
        print("T=",T)
        print("Loss:",np.sum((x.dot(w) - y)**2)/ x.shape[0] /2) #最小二乘损失
        print((x.dot(w) - y)**2)
    gradient = 2 * x.T.dot(x.dot(w)-y) #损失的导数x*(yh-h)
    adagrad_sum += gradient ** 2
    w = w - learning_rate * gradient / (np.sqrt(adagrad_sum) + 0.0005)

np.save('weight.npy',w)
w = np.load('weight.npy')
test_raw_data = np.genfromtxt("test.csv", delimiter=',')   ## test.csv
test_data = test_raw_data[:, 2: ]
where_are_NaNs = np.isnan(test_data)
test_data[where_are_NaNs] = 0 

test_x = np.empty(shape = (240, 18 * 9),dtype = float)

for i in range(240):
    test_x[i,:] = test_data[18 * i : 18 * (i+1),:].reshape(1,-1) 

for i in range(test_x.shape[0]):        ##Normalization
    for j in range(test_x.shape[1]):
        if not std[j] == 0 :
            test_x[i][j] = (test_x[i][j]- mean[j]) / std[j]

test_x = np.concatenate((np.ones(shape = (test_x.shape[0],1)),test_x),axis = 1).astype(float)
answer = test_x.dot(w)

f = open("result.csv","w")
w = csv.writer(f)
title = ['id','value']
w.writerow(title) 
for i in range(240):
    content = ['id_'+str(i),answer[i][0]]
    w.writerow(content)