import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math

data = np.loadtxt('Apr-May_mod.csv',delimiter=",")   # 사용할 데이터 불러오기
set_num = 300 #몇세트 뽑을지 정하기

X = []
Y = []
index = np.arange(1,60).reshape(59,1)
Pmax = np.amax(data[:,1:4])
Vmax = np.amax(data[:,4])


for i in range(0,set_num):  # 60분씩 100세트 추출하여 1~59분까지 input X, 60분이 ouput Y인 리스트만들기 
    start_point = random.randrange(1,len(data)-60)
    X.append(np.hstack((index, data[start_point:start_point+59,:])))
    Y.append(np.hstack(([60], data[start_point+60,:])))

X = torch.tensor(X, dtype= torch.float) # 그 데이터를 텐서화하기
Y = torch.tensor(Y, dtype= torch.float)


X[:,:,0] = torch.div(X[:,:,0],60)
X[:,:,1:5] = torch.div(X[:,:,1:5], Pmax)
X[:,:,5] = torch.div(X[:,:,5], Vmax)

Y[:,0] = torch.div(Y[:,0],60)
Y[:,1:5] = torch.div(Y[:,1:5], Pmax)
Y[:,5] = torch.div(Y[:,5], Vmax)


X = X.view([set_num,-1])    # X를 2차원으로
# Y = Y.unsqueeze(1)



class NeuralNetwork(nn.Module):
    def __init__(self):
        self.inputSize = 354
        self.outputSize = 6
        self.hiddenSize = 500


        self.w1 = torch.randn(self.inputSize, self.hiddenSize, requires_grad=True)
        # self.w2 = torch.randn(self.hiddenSize, self.outputSize, requires_grad=True)
        self.w2 = torch.randn(self.hiddenSize, self.hiddenSize, requires_grad=True)
        self.w3 = torch.randn(self.hiddenSize, self.outputSize, requires_grad=True)

        self.learning_rate = 50

    def forward(self,X):
        # self.z1 = torch.matmul(X, self.w1)
        # self.z2 = torch.sigmoid(self.z1)
        # self.z3 = torch.matmul(self.z2, self.w2)
        # out = torch.sigmoid(self.z3)
        self.z1 = torch.matmul(X, self.w1)
        self.z2 = torch.sigmoid(self.z1)
        self.z3 = torch.matmul(self.z2, self.w2)
        self.z4 = torch.sigmoid(self.z3)
        self.z5 = torch.matmul(self.z4, self.w3) 
        out = torch.sigmoid(self.z5)
        # out = self.z5 
        return out
    
    def train(self, X, Y):
        out = self.forward(X)
        # print(Y.shape)
        # print(out.shape)
        # m = Y.shape[0]
        # logprobs = torch.multiply(torch.log(out),Y) + torch.multiply(torch.log(1-out),1-Y)
        # error = -1/m * torch.sum(logprobs)
        error = ((Y-out)**2).mean()
        error.backward()
        self.w1.data -= self.learning_rate * (self.w1).grad #여기를 +=에서 수정함
        self.w2.data -= self.learning_rate * (self.w2).grad
        self.w3.data -= self.learning_rate * (self.w3).grad
        self.w1.grad = None #여기를 추가함
        self.w2.grad = None
        self.w3.grad = None
        

    def saveWeights(self, model):
        torch.save(model, "NeuralNetwork")

    def predict(self):
        print("Input Data: ",str(X[1].view([59,6])))
        print("Predicted: ", str(self.forward(X[1]).detach()))


NN = NeuralNetwork()
trainIdx = 2000

for idx in range(trainIdx):
    if idx%100 == 0:
        print("#" + str(idx) + " Loss: " + str(torch.mean((Y-NN.forward(X))**2).detach().item()))
    NN.train(X,Y)
    plt.plot(idx,torch.mean((Y-NN.forward(X))**2).detach().item(), 'ro')

NN.saveWeights(NN)
NN.predict()
plt.show()
