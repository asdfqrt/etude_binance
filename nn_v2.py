from datetime import datetime
import numpy as np
import random
from pandas.core.indexes.datetimes import DatetimeIndex
from pandas.core.tools.datetimes import DatetimeScalar
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import pandas as pd
import mplfinance as mpf

data = pd.read_csv('Jun-Jun_1M_Mod2.csv',delimiter=",") #pandas로 불러오는게 numpy의 loadtxt보다 훨씬 빠르다
data = data.to_numpy()
train_set_num = 1000 #몇세트 뽑을지 정하기

X = []
Y = []

# Pmax = np.amax(data[:,1:4])
# Vmax = np.amax(data[:,4])
# rise_rate = 1.0

"""
트레이닝 셋
"""
for i in range(0,train_set_num):  
    start_point = random.randrange(1,len(data)-60)
    X.append(data[start_point:start_point+55,:])
    Y.append(data[start_point+55:start_point+60,:])


# for i in range(0,train_set_num):  
#     start_point = random.randrange(1,len(data)-240)
#     X.append(data[start_point:start_point+180,:])
#     if np.mean(data[start_point+181:start_point+240,3]) > rise_rate * data[start_point+180,3]:
#         Y.append(1)
#     else:
#         Y.append(0)
        

# print("평균 : " + str(np.mean(Y)))

X = torch.tensor(X, dtype= torch.float).cuda() # 그 데이터를 텐서화하기
Y = torch.tensor(Y, dtype= torch.float).cuda()

# X[:,:,0:4] = torch.div(X[:,:,0:4], Pmax)
# X[:,:,4] = torch.div(X[:,:,4], Vmax)

# Y[:,0:4] = torch.div(Y[:,0:4], Pmax)
# Y[:,4] = torch.div(Y[:,4], Vmax)



X = X.view([train_set_num,-1])    # X를 2차원으로
Y = Y.view(train_set_num,-1)

"""
테스트 셋
"""
test_set_num = 1000

X_test = []
Y_test = []

for i in range(0,test_set_num):  
    start_point = random.randrange(1,len(data)-60)
    X_test.append(data[start_point:start_point+55,:])
    Y_test.append(data[start_point+55:start_point+60,:])


# for i in range(0,test_set_num):  
#     start_point = random.randrange(1,len(data)-240)
#     X_test.append(data[start_point:start_point+180,:])
#     if np.mean(data[start_point+181:start_point+240,3]) > rise_rate * data[start_point+180,3]:
#         Y_test.append(1)
#     else:
#         Y_test.append(0)


X_test = torch.tensor(X_test, dtype= torch.float).cuda() 
Y_test = torch.tensor(Y_test, dtype= torch.float).cuda()

# X_test[:,:,0:4] = torch.div(X_test[:,:,0:4], Pmax)
# X_test[:,:,4] = torch.div(X_test[:,:,4], Vmax)

# Y_test[:,0:4] = torch.div(Y_test[:,0:4], Pmax)
# Y_test[:,4] = torch.div(Y_test[:,4], Vmax)



X_test = X_test.view([test_set_num,-1])   
Y_test = Y_test.view(test_set_num,-1)






class NeuralNetwork(nn.Module):
    def __init__(self):

        super(NeuralNetwork, self).__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(4*55, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300, 300),
            nn.ReLU(),
            nn.Linear(300,4*5)
            
            # nn.Linear(300, 1),
            # nn.Sigmoid()
        )

        # self.learning_rate = 5e-4
        self.learning_rate = 1e-4
        self.optimizer = torch.optim.Adam(self.parameters(),lr=self.learning_rate)



    def forward(self,X):
        logits = self.linear_relu_stack(X)
        return logits


    def train(self, X, Y):
        pred = self.forward(X)
        loss_fn = nn.MSELoss()
        loss = loss_fn(pred, Y)
        # self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

    def predict(self, X, Y):

        # print("Input Data: ",str(X[1].view([-1,4])))
        # print("Predicted: ", str(self.forward(X[1]).view([-1,4]).detach()))
        # print("Real Data: ",str(Y[1].view([-1,4])))


        columns = ["Open","High","Low","Close"]
        dates = ["2019-06-07", "2019-06-05", "2019-06-04", "2019-06-03", "2019-05-31"]
        index = pd.to_datetime(dates)
        PredictDataplot = pd.DataFrame(self.forward(X[1]).view([-1,4]).cpu().detach().numpy(), index = index, columns = columns)
        Realdataplot = pd.DataFrame(Y[1].view([-1,4]).cpu().detach().numpy(), index = index, columns = columns)

        fig, (ax1, ax2) = plt.subplots(1,2)
        fig.suptitle('Predict vs Real')
        mpf.plot(PredictDataplot, type='candle',ax=ax1)
        mpf.plot(Realdataplot, type='candle',ax=ax2)

        size_prediction = torch.zeros((len(X),1)).cuda()
        center_prediction = torch.zeros((len(X),1)).cuda()
        size_Y = torch.sub(torch.amax(Y,1),torch.amin(Y,1)) #엄밀하게 따지면 캔들의 중심이 아님,선부분 포함해버림
        center_Y = torch.mean(Y,1,keepdim=True)    #엄밀하게 따지면 센터가 아님

        for i in range(len(X)):
            max_height = max(self.forward(X[i]))
            min_height = min(self.forward(X[i]))
            size_prediction[i,0] = max_height - min_height

            center_prediction[i,0] = torch.mean(self.forward(X[i])) 

        print("Real candle size is bigger than Predicted candle size about : {} %".format(100*torch.mean(torch.div(size_Y,size_prediction))))
        print("Avg center missmatch  : {} ".format(torch.mean(torch.abs(center_Y - center_prediction))))




        # Y_prediction = torch.zeros((len(X),1)).cuda()
        # for i in range(len(X)):
        #     if self.forward(X[i]) > 0.5:
        #         Y_prediction[i,0] = 1
        #     else:
        #         Y_prediction[i,0] = 0

        # print("Accuracy : {} %".format(100-100*torch.mean(torch.abs(Y_prediction - Y))))






NN = NeuralNetwork().cuda()
trainIdx = 2000

for idx in range(trainIdx):
    if idx%100 == 0:
        print("#" + str(idx) + " Loss: " + str(torch.mean((Y-NN.forward(X))**2).detach().item()))
    NN.train(X,Y)
    plt.plot(idx,torch.mean((Y-NN.forward(X))**2).detach().item(), 'ro')
    plt.plot(idx,torch.mean((Y_test-NN.forward(X_test))**2).detach().item(), 'bo')

NN.predict(X,Y)
NN.predict(X_test,Y_test)
plt.show()
