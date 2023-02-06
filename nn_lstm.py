import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from datetime import datetime
import matplotlib.pyplot as plt


"""
LSTM으로 하되 Y값은 매수타이밍이 맞는지를 확인한다
X값으로 10일간의 분봉를 넣고 LSTM으로 각각의 분봉마다 Y 값이 매수타이밍이 맞으면 1 아니면 0으로 간다
Y값은 수동으로 입력(ex) 이후 몇분동안 몇%이상 상승 등등
1분단위 10일간격으로 X 넣고 
many to many 로 연결시키기
순서상 최초의 몇개의 x는 무조건 매수타이밍이 아니게(AI에서 지켜보고 판단하도록) 수동으로 설정
그럴필요 없음 어짜피 초반부는 소프트 맥스 함수에서 3개 분류시 관망으로 나오게될것(many to many 에서 첫부분 the만 나오는것과 같음 아니 이건 입력되서 그런건데)

"""

"""
LSTM으로 하되 Y값은 1분뒤 분봉이다
X값을 1년간의 분봉을 넣고 LSTM으로 각각의 분봉마다 Y값을 1분후 분봉으로 잡는다
Y값은 X값에서 1틱 후값
many to many로 연결시키기

"""
# print(datetime.fromtimestamp(int("1614556800000")/1000))
data = pd.read_csv('Jun-Jun_1M_Mod2.csv',delimiter=",") #pandas로 불러오는게 numpy의 loadtxt보다 훨씬 빠르다
data = data.to_numpy()

# """
# 트레이닝 셋
# """
train_set_num = 1 #몇세트 뽑을지 정하기
# X = np.zeros([2,1])
# Y = np.ones([2,1])
# X = np.concatenate((X,Y),axis=1)
X=[]
Y=[]
for i in range(0,train_set_num):  
    start_point = random.randrange(1,len(data)-200)
    X.append(data[start_point:start_point+200,:])
    Y.append(data[start_point:start_point+200,3])
X = np.array(X)
Y = np.array(Y)
X = X.swapaxes(0,1)
Y = Y.swapaxes(0,1)

X = torch.tensor(X, dtype= torch.float).cuda()
Y = torch.tensor(Y, dtype= torch.float).cuda()

"""
클래스 디자인
"""
class LSTM1(nn.Module):
    def __init__(self, num_classes, input_size, hidden_size, num_layers, seq_length):
        super(LSTM1, self).__init__()
        self.num_classes = num_classes #number of classes
        self.num_layers = num_layers #number of layers
        self.input_size = input_size #input size
        self.hidden_size = hidden_size #hidden state
        self.seq_length = seq_length #sequence length

        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                          num_layers=num_layers, batch_first=True) #lstm
        self.fc_1 =  nn.Linear(hidden_size, 128) #fully connected 1
        self.fc = nn.Linear(128, num_classes) #fully connected last layer

        self.relu = nn.ReLU()

    def forward(self,x):
        h_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #hidden state
        c_0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).cuda() #internal state
        # Propagate input through LSTM
        output, (hn, cn) = self.lstm(x, (h_0, c_0)) #lstm with input, hidden, and internal state
        hn = hn.view(-1, self.hidden_size) #reshaping the data for Dense layer next
        out = self.relu(hn)
        out = self.fc_1(out) #first Dense
        out = self.relu(out) #relu
        out = self.fc(out) #Final Output
        return out
"""
파라미터
"""
num_epochs = 1000 #1000 epochs
learning_rate = 0.1 #0.001 lr
input_size = 4 #number of features
hidden_size = 2 #number of features in hidden state
num_layers = 1 #number of stacked lstm layers
num_classes = 1 #number of output classes 

"""
모델선언
"""
lstm1 = LSTM1(num_classes, input_size, hidden_size, num_layers, X.shape[1]).cuda()
criterion = torch.nn.MSELoss()    # mean-squared error for regression
optimizer = torch.optim.Adam(lstm1.parameters(), lr=learning_rate) 

"""
학습하기
"""
for epoch in range(num_epochs):
  outputs = lstm1.forward(X) #forward pass
  optimizer.zero_grad() #caluclate the gradient, manually setting to 0
 
  # obtain the loss function
  loss = criterion(outputs, Y)
 
  loss.backward() #calculates the loss of the loss function
 
  optimizer.step() #improve from loss, i.e backprop
  if epoch % 100 == 0:
    print("Epoch: %d, loss: %1.5f" % (epoch, loss.item())) 

train_predict = lstm1(X)
data_predict = train_predict.data.cpu().numpy()
actual = Y.data.cpu().numpy()
plt.plot(data_predict, label='Predicted Data') 
plt.plot(actual, label="Actual data")
plt.legend()
plt.show()