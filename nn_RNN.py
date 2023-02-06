import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


df = pd.read_csv('July-July_1H_df.csv')

scaler = MinMaxScaler()
df[['Open','High','Low','Close','Volume']] = scaler.fit_transform(df[['Open','High','Low','Close','Volume']])

X = df[['Open','High','Low','Close','Volume']].values
y = df[['Close']].values


def seq_data(X, y, sequence_length):
    x_seq = []
    y_seq = []

    for i in range(len(X)-sequence_length-target_time):
        x_seq.append(X[i: i+sequence_length])
        y_seq.append(y[i+sequence_length+target_time])

    return torch.FloatTensor(x_seq).to(device), torch.FloatTensor(y_seq).to(device)


split = 6000
sequence_length = 24
target_time = 0

x_seq, y_seq = seq_data(X, y, sequence_length)

x_train_seq = x_seq[:split]
y_train_seq = y_seq[:split]
x_test_seq = x_seq[split:]
y_test_seq = y_seq[split:]

# print(x_train_seq.size(), y_train_seq.size())
# print(x_test_seq.size(), y_test_seq.size())


train = torch.utils.data.TensorDataset(x_train_seq, y_train_seq)
test = torch.utils.data.TensorDataset(x_test_seq, y_test_seq)

batch_size = 20
train_loader = torch.utils.data.DataLoader(dataset=train, batch_size=batch_size, shuffle=False)
test_loader = torch.utils.data.DataLoader(dataset=test, batch_size=batch_size, shuffle=False)

input_size = x_seq.size(2)
num_layers = 2
hidden_size = 8




class VanillaRNN(nn.Module):

    def __init__(self, input_size, hidden_size, sequence_length, num_layers, device):
        super(VanillaRNN, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(nn.Linear(hidden_size * sequence_length, 1),
                                nn.Sigmoid())

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size()[0], self.hidden_size).to(self.device)
        out, _ = self.rnn(x, h0)
        out = out.reshape(out.shape[0], -1)
        out = self.fc(out)
        return out


model = VanillaRNN(input_size=input_size,
                    hidden_size=hidden_size,
                    sequence_length=sequence_length,
                    num_layers=num_layers,
                    device=device).to(device)


criterion = nn.MSELoss()
lr = 1e-3

num_epochs = 200
optimizer = optim.Adam(model.parameters(), lr=lr)


loss_graph = []
n = len(train_loader)

for epoch in range(num_epochs):
    running_loss = 0.0

    for data in train_loader:
        seq, target = data
        out = model(seq)
        loss = criterion(out, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    loss_graph.append(running_loss / n)
    if epoch % 10 == 0:
        print('[epoch: %d] loss: %.6f' %(epoch, running_loss/n))


# plt.figure(figsize=(20,10))
# plt.plot(loss_graph)
# plt.show()

def accuracy(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []
        start_point = [[0]] * sequence_length
        target_point = start_point + [[0]] * target_time

        previous_data = 0
        accuracy_counter = []
        trend_counter = []
        data_num = 0

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()
        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()

        pred = target_point + train_pred + test_pred

        for data in actual:
            if data_num > split:
                try:
                    if previous_data < data and pred[data_num] < pred[data_num+1]:   #실제 데이터 상승세이면서 예측 데이터도 상승세이면
                        accuracy_counter += [1]   #정확도 리스트에 1추가
                    elif previous_data > data and pred[data_num] > pred[data_num+1]: #실제 데이터 하락세이면서 예측 데이터도 하락세이면
                        accuracy_counter += [1]
                    else:
                        accuracy_counter += [0]
                except:
                    pass                    # 실제 데이터보다 예측데이터가 길이가 짧기 때문에 오류발생

                try:
                    present_pred =np.array(pred[data_num],dtype=float)
                    if abs(previous_data - present_pred) > abs(data-present_pred): #이전 예측값에 가까워지도록 실제값이 이동했으면
                        trend_counter += [1]
                    else:
                        trend_counter += [0]
                except:
                    pass
            previous_data = data
            data_num = data_num +1
        accuracy = sum(accuracy_counter) / len(accuracy_counter)
        trend = sum(trend_counter) / len(trend_counter)
    return accuracy, trend
print('accuracy: %.4f, trend following: %.4f' %accuracy(train_loader, test_loader, df['Close']))


def plotting(train_loader, test_loader, actual):
    with torch.no_grad():
        train_pred = []
        test_pred = []
        start_point = [[0]] * sequence_length
        target_point = start_point + [[0]] * target_time

        for data in train_loader:
            seq, target = data
            out = model(seq)
            train_pred += out.cpu().numpy().tolist()

        for data in test_loader:
            seq, target = data
            out = model(seq)
            test_pred += out.cpu().numpy().tolist()

    # total = start_point + train_pred + test_pred
    total2 = target_point + train_pred + test_pred

    plt.figure(figsize=(20,10))
    plt.plot(np.ones(100)*len(train_pred), np.linspace(0,1,100), '--', linewidth=0.6)
    plt.plot(actual, '--')
    # plt.plot(total, '-',linewidth=0.6)
    plt.plot(total2, '-.')

    plt.legend(['train boundary', 'actual', 'real-time prediction'])
    plt.title('predicted %d hours earlier' %(target_time+1),loc='right',pad=20)
    plt.xlabel('time')
    plt.ylabel('Close')
    plt.show()
plotting(train_loader, test_loader, df['Close'])



"""
현재 문제가 되는건 
예측하기위한 x_seq, y_seq가 현재 존재하는 X,y값에서 50시간 미만인 경우에 시작점이 있어야 에측이 가능함
즉 지금 이미 나온 가격까지만 예측가능하다

일례로 리얼타임에서 20시간 이전 지점에서 예측을 하려고 할경우 50시간의 데이터가 없기때문에(20시간치밖에 없다)
X,y데이터가 부족하게됨
본디 RNN에선 이렇게 데이터 길이가 바뀌어도 예측이 가능해야되는데 지금 이 코드의 경우엔 일정하게 50시간 간격으로만 받고있음
거기서 오류가 발생할것


내생각에 test_seq말고 최후에 할때는 whole_seq라고 해서 y값없는 부분 끊지말고 전부 예측시키면 될거 같은데



그리고 이코드엔 문제가 있는데
그래프로 표시될때 +50시간에서의 가격 예측을 한걸 0에다가 띄워줌
이거 고쳐야됨 == 수정완료


문제 하나 더
실질적으로 가격을 예측하는것이아닌
당시의 마지막가격을 그대로 찍는것이 loss를 제일 줄일수잇는방법이라고 알고리즘이 판단하는듯?



상승한다고 예측해놓고
실제로 상승했는지, 하락한다고 예측해놓고 실제로 하락했는지의 비율




"""