import numpy as np
import matplotlib.pyplot as plt

def _normalize_column_0_1(X, train=True, specified_column = None, X_min = None, X_max=None):
    # The output of the function will make the specified column of the training data 
    # from 0 to 1
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the max value of the 
    # training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_max = np.reshape(np.max(X[:, specified_column], 0), (1, length))
        X_min = np.reshape(np.min(X[:, specified_column], 0), (1, length))
        
    X[:, specified_column] = np.divide(np.subtract(X[:, specified_column], X_min), np.subtract(X_max, X_min))
    return X

def _normalize_column_normal(X, train=True, specified_column = None, X_mean=None, X_std=None):
    # The output of the function will make the specified column number to 
    # become a Normal distribution
    # When processing testing data, we need to normalize by the value 
    # we used for processing training, so we must save the mean value and 
    # the variance of the training data
    if train:
        if specified_column == None:
            specified_column = np.arange(X.shape[1])
        length = len(specified_column)
        X_mean = np.reshape(np.mean(X[:, specified_column],0), (1, length))
        X_std  = np.reshape(np.std(X[:, specified_column], 0), (1, length))
    
    X[:,specified_column] = np.divide(np.subtract(X[:,specified_column],X_mean), X_std) 
    return X

def _shuffle(X, Y):
    randomize = np.arange(len(X))
    np.random.shuffle(randomize)
    return (X[randomize], Y[randomize])
    
def train_test_split(X, y, test_size=0.25):
    train_len = int(round(len(X)*(1-test_size)))
    return X[0:train_len], y[0:train_len], X[train_len:None], y[train_len:None]

def _sigmoid(z):
    # sigmoid function can be used to output probability
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-6, 1-1e-6)

def get_prob(X, w, b):
    # the probability to output 1
    return _sigmoid(np.add(np.matmul(X, w), b))

def infer(X, w, b):
    # use round to infer the result
    return np.round(get_prob(X, w, b))

def _cross_entropy(y_pred, Y_label):
    # compute the cross entropy
    cross_entropy = -np.dot(Y_label, np.log(y_pred))-np.dot((1-Y_label), np.log(1-y_pred))
    return cross_entropy

def _gradient(X, Y_label, w, b):
    # return the mean of the graident
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad

def _gradient_regularization(X, Y_label, w, b, lamda):
    # return the mean of the graident
    y_pred = get_prob(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.mean(np.multiply(pred_error.T, X.T), 1)+lamda*w
    b_grad = -np.mean(pred_error)
    return w_grad, b_grad

def _loss(y_pred, Y_label, lamda, w):
    return _cross_entropy(y_pred, Y_label) + lamda * np.sum(np.square(w))

def accuracy(Y_pred, Y_label):
    acc = np.sum(Y_pred == Y_label)/len(Y_pred)
    return acc

x = np.genfromtxt('X_train',delimiter=',',skip_header=1)  #32561* 106
y = np.genfromtxt('Y_train',delimiter=',',skip_header=1)
test_size = 0.1155
x, y, x_test, y_test = train_test_split(x, y, test_size = test_size)
col = [0,1,3,4,5,7,10,12,25,26,27,28]
x = _normalize_column_0_1(x, specified_column=col)



# Use 0 + 0*x1 + 0*x2 + ... for weight initialization
w = np.zeros((x.shape[1],)) 
b = np.zeros((1,))
lamda = 0.001

max_iter = 40  # max iteration number
batch_size = 32 # number to feed in the model for average to avoid bias
learning_rate = 0.2  # how much the model learn for each step
num_train = len(y)
num_test = len(y_test)
step =1
loss_train = []
loss_validation = []
train_acc = []
test_acc = []

for epoch in range(max_iter):
    # Logistic regression train with batch
    for idx in range(int(np.floor(len(y)/batch_size))):
        x_bt = x[idx*batch_size:(idx+1)*batch_size]
        y_bt = y[idx*batch_size:(idx+1)*batch_size]
        
        # Find out the gradient of the loss
        y_bt_pred = get_prob(x_bt, w, b)
        pred_error = y_bt - y_bt_pred
        w_grad = -np.mean(np.multiply(pred_error, x_bt.T), 1)+lamda*w
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
    loss_train.append(_loss(y_pred, y, lamda, w)/num_train)
    
    y_test_pred = get_prob(x_test, w, b)
    yh_test = np.round(y_test_pred)
    test_acc.append(accuracy(yh_test, y_test))
    loss_validation.append(_loss(y_test_pred, y_test, lamda, w)/num_test)

plt.plot(loss_train)
plt.plot(loss_validation)
plt.legend(['train', 'test'])
plt.show()

plt.plot(train_acc)
plt.plot(test_acc)
plt.legend(['train', 'test'])
plt.show()
