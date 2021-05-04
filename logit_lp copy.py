import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score

num_epochs = 20
batch_size = 100
learning_rate = 0.000001

data = pd.read_csv("USAir.csv")


train_data = np.array(data.iloc[:,0:8])
input_size = np.size(train_data,1)
print(train_data.shape)




