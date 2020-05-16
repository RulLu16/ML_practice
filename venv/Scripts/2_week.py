import numpy as np

x_dim=2
lr=0.01

def generate_toy_data(size):
    '''
        generates the toy data consisting of positive(50%) and negative(50%) examples
          positive mean : [1,1]
          negative mean : [-1,-1]
        Arguments:
          size -- number of positive examples(or negative examples)
        Returns:
          the tuple [feature vectors,labels]
      '''
    x0=np.random.normal(size=2*size).reshape(-1,2)-1
    x1=np.random.normal(size=2*size).reshape(-1,2)+1
    return np.concatenate([x0,x1]), np.concatenate([np.zeros(size), np.ones(size)]).astype(np.int)

train_x, train_y=generate_toy_data(30)
test_x, test_y=generate_toy_data(10)

print(test_x)
print(test_y)

def sigmoid(z):
    return 1/(1+np.e**(-z))

def forward(w,b,X):
  """
    computes the logistic regression value for the given input
    Arguments:
      w -- weights of size [x_dim]
      b -- bias, a scalar
      X -- input data of size [batch, x_dim]
    Returns:
      sigmoid(z) -- probabilities of being in class 1 of size [batch]
  """
  z = b + np.matmul(w,X.T)
  return sigmoid(z)

def loss(y,y_hat):
    return np.sum((y-y_hat)**2/len(y))

def optimize(w,b,X,y,y_hat,lr=0.01):
    """
        optimize one step
        Arguments:
          w -- weights of size [x_dim]
          b -- bias, a scalar
          X -- input data of size [batch, x_dim]
          y -- labels of size [batch]
          y_hat -- predictions of size [batch]
          lr -- learning rate
        Returns:
          w -- optimized weights
          b -- optimized bias
      """
    bsize=len(y)

    dw=np.zeros_like(w)
    for i in range(len(dw)):
        dw[i]=0

    db=0

    w=w-lr*dw
    b=b-lr*db

    return w,b

def train(w,b,X,y,epoch,lr=0.001):
    for _ in range(epoch):
        y_hat=forward(w,b,X)
        print(loss(y,y_hat))
        w,b=optimize(w,b,X,y,y_hat,lr=lr)

    return w,b

def eval(w,b,X,y):
    y_hat=forward(w,b,X)
    res=[0 if i<0.5 else 1 for i in y_hat]
    res=(res==y)
    res=np.sum(res)/len(y)
    print("accuracy : "+str(res))
    return res

weights=np.zeros(x_dim)
bias=0
print(weights, bias)
eval(weights, bias, test_x, test_y)

weight, bias = train(weights, bias, train_x, train_y,100,lr=0.2)
print(weights,bias)


