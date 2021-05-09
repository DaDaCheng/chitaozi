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
from torch import linalg as LA


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


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

fname ='neural2'
#fname ='/content/drive/My Drive/pan/USAir'
n = 297
batch_size = 100
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



from torch_geometric.nn import MessagePassing

class WRNN(torch.nn.Module):
    def __init__(self,input_size):
        super(WRNN, self).__init__()
        self.conv1 = GCNConv_nonorm(input_size, input_size)
    def forward(self, x, edge_index):
        h = self.conv1(x, edge_index)
        return h

class GCNConv_nonorm(MessagePassing):
    def __init__(self, in_channels, out_channels,diag=False):
        super(GCNConv_nonorm, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.diag=diag
        if self.diag:
            self.diagele=nn.Parameter(torch.ones(in_channels).clone().detach()).to(device)
            self.mat=torch.diag(self.diagele)
        else:
            self.lin = torch.nn.Linear(in_channels, out_channels,bias=False)
            self.lin.weight = nn.Parameter(torch.eye(in_channels).clone().detach())

    def forward(self, x, edge_index):
        if self.diag:
            x= torch.matmul(x,self.mat)
        else:
            x = self.lin(x)
            


        
        norm = torch.ones(edge_index.size(1),1).to(device)
        return self.propagate(edge_index, x=x, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1, 1) * x_j

class MLP(torch.nn.Module):
    def __init__(self,input_size):
        super(MLP, self).__init__()
        self.linear1 = torch.nn.Linear(input_size,200)
        self.linear2 = torch.nn.Linear(200,100)
        self.linear3 = torch.nn.Linear(100,20)
        self.linear4 = torch.nn.Linear(20,1)
        self.act1= nn.ReLU()
        self.act2= nn.ReLU()
        self.act3= nn.ReLU()

    def forward(self, x):
        out= self.linear1(x)
        out = self.act1(out)
        out= self.linear2(out)
        out = self.act2(out)
        out = self.linear3(out)
        out = self.act3(out)
        out = self.linear4(out)
        out = torch.sigmoid(out)
        return out


class GraphNet(torch.nn.Module):
    def __init__(self, hidden_channels,walk_len=7):
        super(GraphNet, self).__init__()
        #self.lin = torch.nn.Linear(non_perturb_train.num_node_features,hidden_channels)
        self.wrnn = WRNN(hidden_channels)
        self.walk_len = walk_len
    def forward(self, x_p, x_np, y, edge_index_p,edge_index_np):
        #h_p = x_p
        #h_np = x_np
        h_p = self.wrnn(x_p,edge_index_p)
        h_np = self.wrnn(x_np,edge_index_np)
        scale_factor = LA.norm(h_p,dim=0)
        scale_factor = scale_factor[0]
        h_p = h_p/scale_factor
        h_np = h_np/scale_factor
        h_p = self.wrnn(h_p,edge_index_p)
        h_np = self.wrnn(h_np,edge_index_np)
        h_p = h_p/scale_factor
        h_np = h_np/scale_factor
        batch_size = y.size(0)

        p_list = torch.zeros(batch_size,self.walk_len)
        np_list = torch.zeros(1,self.walk_len)
        
        for i in range(0,self.walk_len):
            h_p = self.wrnn(h_p,edge_index_p)
            h_np = self.wrnn(h_np,edge_index_np)
            h_p = h_p/scale_factor
            h_np = h_np/scale_factor
            #h_p = h_p.relu()
            #h_np = h_np.relu()
            val = torch.trace(h_np)
            #np_list[0,i] =  torch.sign(val)*torch.log(torch.abs(val))
            np_list[0,i] = val
            for j in range(batch_size):
                val = torch.trace(h_p[j*n:j*n+n,:])
                #p_list[j,i] = torch.sign(val)*torch.log(torch.abs(val))
                #p_list[j,i] = torch.sign(val)*torch.log(torch.abs(val))
                p_list[j,i] = val
        np_list = np_list.repeat(batch_size,1)
        p_list = p_list- np_list
        p_list = p_list.to(device)
        p_list=p_list*100
        for i in range(0,batch_size):
            p_list[i,:] = torch.mul(p_list[i,:],(y[i,0]-0.5)*2)
        mu = torch.mean(p_list,dim=0,keepdim=False)
        std = torch.std(p_list,dim=0,keepdim=False)
        p_list = (p_list-mu)/std
        #print('P_list,shape',p_list.shape)
        return p_list


IfTrainGraph = True
walk_len=8
Graphmodel = GraphNet(hidden_channels=n,walk_len=walk_len).to(device)
graphout=[]
datalist=[]



for step, data in enumerate(train_loader):
    data = data.to(device)
    datalist.append(data)
    with torch.no_grad():
        out = Graphmodel(data.x, non_perturb_train.x, data.y, data.edge_index, non_perturb_train.edge_index)
        graphout.append(out.clone().detach())
        #graphout.append(torch.randn_like(out))
criterion = torch.nn.BCELoss()
#criterion =torch.nn.MSELoss()
mlpmodel=MLP(walk_len).to(device)

optimizergraph = torch.optim.Adam(Graphmodel.parameters(), lr=0.0001)
#optimizergraph = torch.optim.SGD(Graphmodel.parameters(), lr=-0.001)
optimizermlp = torch.optim.Adam(mlpmodel.parameters(), lr=0.01)



def train(train_loader,IfTrainGraph,IfTrainMLP):
    Graphmodel.train()
    mlpmodel.train()
    loss_eposch=0
    for step, data in enumerate(train_loader):# Iterate in batches over the training dataset.
        data = data.to(device)
        if IfTrainGraph:
            out = Graphmodel(data.x, non_perturb_train.x, data.y, data.edge_index, non_perturb_train.edge_index)
            out = mlpmodel(out)
        else:
            out = mlpmodel(graphout[step])
        target=data.y[:,0]
        out = out.reshape(-1).to(device) # Perform a single forward pass.
        loss = criterion(out, target)  # Compute the loss.
        loss.backward() 
        # if IfTrainMLP:
        #      optimizermlp.zero_grad()
        #      optimizermlp.step()
        # if IfTrainGraph:
        #      optimizergraph.zero_grad()
        #      optimizergraph.step()
        # if step==8:
        #     break





        nloss = -loss.clone()
        if IfTrainMLP:
             optimizermlp.zero_grad()
             loss.backward(retain_graph=True) 
             optimizermlp.step()
             #loss.backward() 
        if IfTrainGraph:
             optimizergraph.zero_grad()
             nloss.backward(retain_graph=True)
             optimizergraph.step()
        if step==8:
            break
             



        #optimizer.step()  # Update parameters based on gradients.
        #optimizer.zero_grad()  # Clear gradients.
        #print(f"Step:{step},Loss:{loss.item()}")
        loss_eposch=loss_eposch+loss.item()
        #print(loss.item())
        
    return loss_eposch/(step+1.0)

def test(loader):
        mlpmodel.eval()
        Graphmodel.eval()
        with torch.no_grad():
            correct = 0
            for step, data in enumerate(loader):  # Iterate in batches over the training/test dataset.
                #print(f'Num of graphs in the current batch: {data.num_graphs}')
                data = data.to(device)
                out = Graphmodel(data.x, non_perturb_train.x, data.y, data.edge_index, non_perturb_train.edge_index)
                out = mlpmodel(out)
                scores = out.cpu().detach().numpy()
                #print(scores)
                labels = data.y[:,1].cpu().detach().numpy()
                break
        return roc_auc_score(labels, scores)  # Derive ratio of correct predictions.

num_epochs = 10000
for epoch in range(num_epochs):
    # if epoch<10:
    #     traing=False
    # else:
    #     traing=True
    loss_step = train(datalist,True, True)
    if epoch%1==0:
        train_acc = test(datalist)
        test_acc = test(test_loader)
    print('Epoch [{}/{}], Loss: {:.4f}, Train AUC: {:.4f}, Test AUC: {:.4f}'\
                    .format(epoch+1, num_epochs, loss_step, train_acc, test_acc))


