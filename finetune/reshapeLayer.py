import torch.nn as nn
from torch.autograd import Variable as Var
#Reshape Layer to transform Tensor into Vector for Fully Connected Layer computation

class ReshapeLayer(nn.Module):
    def __init__(self):
        super(ReshapeLayer, self).__init__()

    def forward(self, X):
        return X.view(X.size(0), -1)