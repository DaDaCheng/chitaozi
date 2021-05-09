import pandas as pd
import numpy as np
import torch
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torch_geometric.data import Data, DataLoader
import torch.nn as nn
from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
GCN_hidden1 = 128;
GCN_hidden2 = 64;
GCN_hidden3 = 32;
GCN_hidden4 = 16;

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
#print(torch.cuda.get_device_name(0))

def non_ptb_data(index, n, x, y):
    # prepare the non-perturbed data
    edge_index = np.concatenate((index,index[:,[1,0]]),axis=0)
    edge_index = torch.tensor(edge_index,dtype=torch.long)
    x = torch.as_tensor(x,dtype=torch.float)
    y = torch.as_tensor(y,dtype=torch.float)
    data = Data(x=x,edge_index=edge_index.t().contiguous(),num_nodes=n, y=y)
    return data

def ptb_data_train(index, n, x, label):
    # prepare the perturbed data with the edge (s,t) added or removed
    # if label == 1 then remove the edge, and add the edge otherwise
    labels = np.zeros((1,2))
    labels[0,0] = label[2]
    labels[0,1] = label[2]
    if label[2] == 0:
        edge = np.zeros((1,2))
        edge[0,0] = label[0]
        edge[0,1] = label[1]
        edge_index = np.concatenate((index,edge),axis=0)
        data = non_ptb_data(edge_index,n,x,labels)
    else:
        ind = np.where((index == (label[0], label[1])).all(axis=1))
        edge_index = np.delete(index,ind[0],0)
        data = non_ptb_data(edge_index,n,x,labels)
    return data

def ptb_data_test(index, n, x, label):
    # prepare the perturbed data with the edge (s,t) added or removed
    # if label == 1 then remove the edge, and add the edge otherwise
    labels = np.zeros((1,2))
    labels[0,0] = 0
    labels[0,1] = label[2]

    edge = np.zeros((1,2))
    edge[0,0] = label[0]
    edge[0,1] = label[1]
    edge_index = np.concatenate((index,edge),axis=0)
    data = non_ptb_data(edge_index,n,x,labels)
    return data

class WRNN(torch.nn.Module):
    def __init__(self,input_size):
        super(WRNN, self).__init__()
        self.conv1 = GCNConv(input_size, input_size)

    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        return h


class MLP(torch.nn.Module):
    def __init__(self,input_size):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size,15)
        self.linear2 = torch.nn.Linear(15,1)
        #self.linear3 = torch.nn.Linear(50,25)
        #self.linear4 = torch.nn.Linear(25,12)
        #self.linear5 = torch.nn.Linear(12,1)
        self.act1= nn.ReLU()
        #self.act2= nn.ReLU()
        #self.act3= nn.ReLU()
        #self.act2= torch.sin
        #self.act3= torch.sin
        #self.act4= nn.ReLU()

    def forward(self, x):
        out= self.linear1(x)
        out = self.act1(out)
        out= self.linear2(out)
        #out = self.act2(out)
        #out = self.linear3(out)
        #out = self.act3(out)
        #out = self.linear4(out)
        #out = self.act4(out)
        #out = self.linear5(out)
        out = torch.sigmoid(out)
        return out

class Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(Net, self).__init__()
        torch.manual_seed(12345)
        self.lin = torch.nn.Linear(non_perturb_train.num_node_features,hidden_channels)
        self.wrnn = WRNN(hidden_channels)
        self.walk_len = 7
        self.mlp = MLP(self.walk_len)
        

    def forward(self, x_p, x_np, y, edge_index_p,edge_index_np):
        #h_p = self.lin(x_p)
        #h_p = h_p.sigmoid()
        #h_np = self.lin(x_np)
        #h_np = h_np.sigmoid()

        h_p = x_p
        h_np = x_np

        batch_size = y.size(0)
        p_list = torch.zeros(batch_size,self.walk_len)
        np_list = torch.zeros(1,self.walk_len)

        for i in range(self.walk_len):
            h_p = self.wrnn(h_p,edge_index_p)
            h_np = self.wrnn(h_np,edge_index_np)
            np_list[0,i] =  torch.trace(h_np)
            for j in range(batch_size):
                p_list[j,i] = torch.trace(h_p[j*n:j*n+n,:])

        np_list = np_list.repeat(batch_size,1)
        p_list = p_list- np_list
        p_list = p_list.to(device)
        for i in range(0,batch_size):
            p_list[i,:] = torch.mul(p_list[i,:],(y[i,0]-0.5)*2)

        mu = torch.mean(p_list,dim=0,keepdim=True)
        std = torch.std(p_list,dim=0,keepdim=True)
        p_list = (p_list-mu)/std
        z = self.mlp(p_list)
        return z


if __name__ == '__main__':
    fname = 'neural2'
    n = 297
    batch_size = 350
    x = torch.eye(n,dtype=torch.float)
    #x = torch.ones((n,1))
    train_data = pd.read_csv(fname+'_train_el.csv',header=None)
    train_edge_index = np.array(train_data.iloc[:,0:])
    non_perturb_train = non_ptb_data(train_edge_index,n,x,np.zeros((1,2))).to(device)

    train_labels = pd.read_csv(fname+'_train_labels.csv',header=None)
    train_labels = np.array(train_labels.iloc[:,0:])
    test_labels = pd.read_csv(fname+'_test_labels.csv',header=None)
    test_labels = np.array(test_labels.iloc[:,0:])

    train_data_list = [ptb_data_train(train_edge_index,n,x,train_labels[i,0:3]) for i in range(0,len(train_labels))]
    train_loader = DataLoader(train_data_list,batch_size=batch_size,shuffle=True)
    test_data_list = [ptb_data_test(train_edge_index,n,x,test_labels[i,0:3]) for i in range(0,len(test_labels))]
    test_loader = DataLoader(test_data_list,batch_size=int(len(test_data_list)),shuffle=True)

    model = Net(hidden_channels=n).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    #criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.BCELoss()
    def train(train_loader):
        model.train()
        loss_eposch=0
        for step, data in enumerate(train_loader):# Iterate in batches over the training dataset.
            data = data.to(device)
            out = model(data.x, non_perturb_train.x, data.y, data.edge_index, non_perturb_train.edge_index) 
            out = out.reshape(-1).to(device) # Perform a single forward pass.
            #print('aa',out)
            #print('bb',data.y[:,0])
            loss = criterion(out, data.y[:,0])  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.
            #print(f"Step:{step},Loss:{loss.item()}")
            loss_eposch=loss_eposch+loss.item()
        return loss_eposch/(step+1.0)

    def test(loader):
         model.eval()

         correct = 0
         for step, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
             #print(f'Num of graphs in the current batch: {data.num_graphs}')
             data = data.to(device)
             out = model(data.x, non_perturb_train.x, data.y, data.edge_index, non_perturb_train.edge_index) 
             scores = out.cpu().detach().numpy()
             #print(scores)
             labels = data.y[:,1].cpu().detach().numpy()
             break
         return roc_auc_score(labels, scores)  # Derive ratio of correct predictions.

    num_epochs = 40
    for epoch in range(0, num_epochs):
        loss_step = train(train_loader)
        train_acc = test(train_loader)
        test_acc = test(test_loader)
        print('Epoch [{}/{}], Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'\
                    .format(epoch+1, num_epochs, loss_step, train_acc, test_acc))