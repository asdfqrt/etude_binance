import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn
import torch.optim as optim

import matplotlib.pyplot as plt
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# df = pd.read_csv('Jan-May_1H_df.csv')
# df = pd.read_csv('Jun-Jun_1M_df.csv')
df = pd.read_csv('May-July_15M_df.csv')

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


split = 20000
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


            previous_data = data
            data_num = data_num +1
        accuracy = sum(accuracy_counter) / len(accuracy_counter)
    return accuracy
print('accuracy: %.4f' %accuracy(train_loader, test_loader, df['Close']))


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


