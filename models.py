import torch
import torch.nn as nn
import torch.nn.functional as F
from lightgbm import LGBMRegressor

class LGBMR():
    pass

class GPR():
    pass

class NeuralNet(nn.Module):
    def __init__(self, n_feature, n_hidden, n_output, p):
        super(NeuralNet, self).__init__()
        self.hidden_1 = nn.Linear(n_feature, n_hidden)
        self.hidden_2 = nn.Linear(n_hidden, n_hidden)
        self.predict = nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = F.relu(self.dropout(self.hidden_1(x)))
        x = F.relu(self.dropout(self.hidden_2(x)))
        x = self.predict(x)
        return x

