import torchvision.models as tvm
import torch.nn as nn


def alex_fc6():
    model = tvm.alexnet(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:3])
    model.classifier = nc
    model.eval()
    return model

def alex_fc7():
    model = tvm.alexnet(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = nc
    model.eval()
    return model
