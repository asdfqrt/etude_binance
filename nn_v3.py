import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
from torch.nn.modules.container import ParameterDict

data = pd.read_csv('Jun-Jun_1M_Mod.csv',delimiter=",") #pandas로 불러오는게 numpy의 loadtxt보다 훨씬 빠르다
checknull = data.isnull().values.any()
assert checknull == False
data = data.to_numpy()
train_set_num = 100000 #몇세트 뽑을지 정하기

X = []
Y = []

Pmax = np.amax(data[:,1:4])
Vmax = np.amax(data[:,4])
rise_rate = 1.0

"""
트레이닝 셋
"""

for i in range(0,train_set_num):  # 60분씩 300세트 추출하여 1~59분까지 input X, 60분이 ouput Y인 리스트만들기 
    start_point = random.randrange(1,len(data)-60)
    # X.append(data[start_point:start_point+59,:])
    # if data[start_point+60,3] > rise_rate * data[start_point+59,3]: #60분째의 종가가 59분째의 종가보다 rise_rate이상 상승한다면
    X.append(data[start_point:start_point+50,:])
    if np.mean(data[start_point+51:start_point+60,3]) > rise_rate * data[start_point+50,3]:
        Y.append(1)
    else:
        Y.append(0)
        

print("평균 : " + str(np.mean(Y)))

X = torch.tensor(X, dtype= torch.float).cuda() # 그 데이터를 텐서화하기
Y = torch.tensor(Y, dtype= torch.float).cuda()

X[:,:,0:4] = torch.div(X[:,:,0:4], Pmax)
X[:,:,4] = torch.div(X[:,:,4], Vmax)

X = X.view([train_set_num,-1])    # X를 2차원으로
Y = Y.view(-1,1)


"""
테스트 셋
"""
# data = np.loadtxt('Mar-Apr_1M_Mod.csv',delimiter=",")
test_set_num = 1000

X_test = []
Y_test = []


for i in range(0,test_set_num):  
    start_point = random.randrange(1,len(data)-60)
    # X_test.append(data[start_point:start_point+59,:])
    # if data[start_point+60,3] > rise_rate * data[start_point+59,3]:
    X_test.append(data[start_point:start_point+50,:])
    if np.mean(data[start_point+51:start_point+60,3]) > rise_rate * data[start_point+50,3]:
        Y_test.append(1)
    else:
        Y_test.append(0)


X_test = torch.tensor(X_test, dtype= torch.float).cuda() 
Y_test = torch.tensor(Y_test, dtype= torch.float).cuda()

X_test[:,:,0:4] = torch.div(X_test[:,:,0:4], Pmax)
X_test[:,:,4] = torch.div(X_test[:,:,4], Vmax)

X_test = X_test.view([test_set_num,-1])   
Y_test = Y_test.view(-1,1)






class NeuralNetwork(nn.Module):
    def __init__(self):
        # self.inputSize = 295
        # self.outputSize = 1
        # self.hiddenSize = 600


        # self.w1 = torch.randn(self.inputSize, self.hiddenSize, device="cuda", requires_grad=True)
        # self.w2 = torch.randn(self.hiddenSize, self.hiddenSize, device="cuda",requires_grad=True)
        # self.w3 = torch.randn(self.hiddenSize, self.hiddenSize, device="cuda",requires_grad=True)
        # self.w4 = torch.randn(self.hiddenSize, self.hiddenSize, device="cuda",requires_grad=True)
        # self.w5 = torch.randn(self.hiddenSize, self.hiddenSize, device="cuda",requires_grad=True)
        # self.w6 = torch.randn(self.hiddenSize, self.outputSize, device="cuda",requires_grad=True)

        # self.learning_rate = 1e-2

        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5*50, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            # nn.Linear(600, 600),
            # nn.ReLU(),
            # nn.Linear(600, 600),
            # nn.ReLU(),
            nn.Linear(300, 1),
            nn.Sigmoid()
        )

        self.learning_rate = 1e-3
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)



    def forward(self,X):
        # self.z1 = torch.matmul(X, self.w1) 
        # self.a1 = torch.sigmoid(self.z1)
        # self.z2 = torch.matmul(self.a1, self.w2) 
        # self.a2 = torch.sigmoid(self.z2)
        # self.z3 = torch.matmul(self.a2, self.w3)
        # self.a3 = torch.sigmoid(self.z3)
        # self.z4 = torch.matmul(self.a3,self.w4)
        # self.a4 = torch.sigmoid(self.z4)
        # self.z5 = torch.matmul(self.a4,self.w5)
        # self.a5 = torch.sigmoid(self.z5)
        # self.z6 = torch.matmul(self.a5, self.w6) 
        # a6 = torch.sigmoid(self.z6)
 

        # return a6
        logits = self.linear_relu_stack(X)
        return logits


    def train(self, X, Y):
        # pred = self.forward(X)
        # m = Y.shape[0]
        # logprobs = torch.multiply(torch.log(pred),Y) + torch.multiply(torch.log(1-pred),1-Y)
        # error = -1/m * torch.sum(logprobs)
        # # loss = nn.MSELoss()
        # # error = loss(out, Y)
        # error.backward()
        # self.w1.data -= self.learning_rate * (self.w1).grad #여기를 +=에서 수정함
        # self.w2.data -= self.learning_rate * (self.w2).grad
        # self.w3.data -= self.learning_rate * (self.w3).grad
        # self.w4.data -= self.learning_rate * (self.w4).grad
        # self.w5.data -= self.learning_rate * (self.w5).grad
        # self.w6.data -= self.learning_rate * (self.w6).grad

        # self.w1.grad = None #여기를 추가함
        # self.w2.grad = None
        # self.w3.grad = None
        # self.w4.grad = None
        # self.w5.grad = None
        # self.w6.grad = None
        pred = self.forward(X)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, Y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def predict(self, X, Y):
        Y_prediction = torch.zeros((len(X),1)).cuda()
        for i in range(len(X)):
            if self.forward(X[i]) > 0.5:
                Y_prediction[i,0] = 1
            else:
                Y_prediction[i,0] = 0

        print("Accuracy : {} %".format(100-100*torch.mean(torch.abs(Y_prediction - Y))))
        # print(Y_prediction-Y)

NN = NeuralNetwork().cuda()
trainIdx = 500

for idx in range(trainIdx):
    if idx%100 == 0:
        print("#" + str(idx) + " Loss: " + str(torch.mean((Y-NN.forward(X))**2).detach().item()))
    NN.train(X,Y)
    plt.plot(idx,torch.mean((Y-NN.forward(X))**2).detach().item(), 'ro')
    plt.plot(idx,torch.mean((Y_test-NN.forward(X_test))**2).detach().item(), 'bo')

NN.predict(X,Y)
NN.predict(X_test,Y_test)
plt.show()
