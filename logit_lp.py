import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

num_epochs = 50
batch_size = 100
learning_rate = 0.0001


data = pd.read_csv("USAir.csv")

train_data = np.array(data.iloc[:,0:8])
input_size = np.size(train_data,1)

train_labels = np.array(data.iloc[:,-2])
test_labels = np.array(data.iloc[:,-1])
pos = np.where((train_labels==0))
test_data = train_data[pos,:]
#test_data = test_data.reshape(train_data.shape[1],train_data.shape[2])
test_labels = test_labels[pos]
test_data = test_data.reshape(np.size(test_data,1),np.size(test_data,2))
print(test_data.dtype)

train_data = torch.from_numpy(train_data.astype(np.float32))
train_labels = torch.from_numpy(train_labels.astype(np.float32))
test_data = torch.from_numpy(test_data.astype(np.float32))
test_labels = torch.from_numpy(test_labels.astype(np.float32))

train_dataset = Data.TensorDataset(train_data, train_labels)
test_dataset = Data.TensorDataset(test_data, test_labels)

train_loader = Data.DataLoader(
    dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True,
)

test_loader = Data.DataLoader(
    dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False,
)

class LogisticRegression(torch.nn.Module):
    def __init__(self,input_size):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_size, 1)     

    def forward(self, x):
        out = torch.sigmoid(self.linear(x))
        return out

model = LogisticRegression(input_size)
criterion = nn.BCELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_step = len(train_loader)
for epoch in range(num_epochs):
    for i, (stats, labels) in enumerate(train_loader):
        stats = stats.reshape(-1, input_size)

        outputs = model(stats)
        outputs = outputs.reshape(-1)
        loss = criterion(outputs,labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i+1)%100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'\
                .format(epoch+1, num_epochs,i+1, total_step, loss.item()))


test_labels = test_labels.detach().numpy()
test_scores = model(test_data).detach().numpy()

auc = roc_auc_score(test_labels, test_scores)
print('Accuracy of the model on the test set: {} %'.format(auc))
print
