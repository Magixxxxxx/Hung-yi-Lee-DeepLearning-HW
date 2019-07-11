import numpy as np
import matplotlib.pyplot as plt

def shuffle(X, Y):
    #打乱X,Y
    randomize = np.arange(len(X)) #怪不得自动生成的是xrange，不同的库用的不太一样，返回的对象也不太一样。py的range()就不支持shuffle
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize]) #ndarray的参数是数组时，返回一个依参数排序后的数组

def train_test_split(X, y, test_size=0.1155):
    #按一个比例分出一部分验证集
    train_len = int(round(len(X)*(1-test_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]

def sigmoid(z):
     # Use np.clip to avoid overflow\超出的部分就把它强置为边界部分
     # e是科学计数法的一种表示 aeN: a*10的N次方
    s = np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)
    return s

def get_prob(X, w, b):
    # the probability to output 1
    return sigmoid(np.add(np.matmul(X, w), b))

def loss(y_pred, Y_label, lamda, w):
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy + lamda * np.sum(np.square(w))

def accuracy(Y_pred, Y_label):
    return np.sum(Y_pred == Y_label)/len(Y_pred)

x = np.genfromtxt('X_train',delimiter=',',skip_header=1)  #32561* 106
y = np.genfromtxt('Y_train',delimiter=',',skip_header=1)

#min max归一
'''
xmin = np.min(x, axis = 0)
xmax = np.max(x, axis = 0)
col = [0,1,3,4,5,7,10,12,25,26,27,28]
x[:,col] = (x[:,col]-xmin[col])/(xmax[col] - xmin[col])#快了0.03-0.01=0.02s
'''
xmean = np.mean(x,axis = 0)
xstd = np.std(x, axis = 0)
col = [0,1,3,4,5,7,10,12,25,26,27,28]
x[:,col] = (x[:,col]-xmean[col])/xstd[col]

x, y, x_test, y_test = train_test_split(x, y)

w = np.zeros(x.shape[1],) #106
b = np.zeros(1,)
lamda = 0.001 #正则化惩罚过拟合
max_iter = 40 #迭代次数
batch_size = 32 # number to feed in the model for average to avoid bias
learning_rate = 0.2
num_train = len(y)
num_dev = len(y_test)
step =1
loss_train = []
loss_validation = []
train_acc = []
test_acc = []

for epoch in range(max_iter):
    # Random shuffle for each epoch
    x, y = shuffle(x, y) #打乱各行数据，这样参数能不易陷入局部最优，模型能够更容易达到收敛。     

    # Logistic regression train with batch
    for idx in range(int(np.floor(len(y)/batch_size))): #每个batch更新一次
        x_bt = x[idx*batch_size:(idx+1)*batch_size] #32*106
        y_bt = y[idx*batch_size:(idx+1)*batch_size] #32*1

        # Find out the gradient of the loss
        y_bt_pred = get_prob(x_bt, w, b) #matmul：二维数组间的dot
        pred_error = y_bt - y_bt_pred
        w_grad = -np.mean(np.multiply(pred_error, x_bt.T), 1)+lamda*w #multiply：数组和矩阵对应位置相乘，输出与相乘数组/矩阵的大小一致
        b_grad = -np.mean(pred_error)

        # gradient descent update
        # learning rate decay with time
        w = w - learning_rate/np.sqrt(step) * w_grad
        b = b - learning_rate/np.sqrt(step) * b_grad
        step = step+1

    # Compute the loss and the accuracy of the training set and the validation set
    y_pred = get_prob(x, w, b)
    yh = np.round(y_pred)
    train_acc.append(accuracy(yh, y))
    loss_train.append(loss(y_pred, y, lamda, w)/num_train)
    
    y_test_pred = get_prob(x_test, w, b)
    yh_test = np.round(y_test_pred)
    test_acc.append(accuracy(yh_test, y_test))
    loss_validation.append(loss(y_test_pred, y_test, lamda, w)/num_dev)

p1 = plt.subplot(121)
p1.plot(loss_train)
p1.plot(loss_validation)
p1.legend(['train', 'test'])
p1.set_title("loss")

p2 = plt.subplot(122)
p2.plot(train_acc)
p2.plot(test_acc)
p2.legend(['train', 'test'])
p2.set_title("accuracy")
plt.show()