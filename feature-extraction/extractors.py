import torchvision.models as tvm
import torch.nn as nn
import torch


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

def vgg19bn_fc6():
    model = tvm.vgg19_bn(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:2])
    model.classifier = nc
    model.eval()
    return model

def vgg19bn_fc7():
    model = tvm.vgg19_bn(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:-2])
    model.classifier = nc
    model.eval()
    return model

def res50_avg():
    model = tvm.resnet50(pretrained=True)
    feats = [None]
    def feat_hook(m, i, o):
        feats[0].copy_(o.data[:, :, 0, 0])
    model._modules.get("avgpool").register_forward_hook(feat_hook)
    def feat_extr(x):
        feats[0] = torch.zeros(x.size(0), 2048)
        model(x)
        return feats[0]
    model.eval()
    return feat_extr

def dense161_last():
    model = tvm.densenet161(pretrained=True)
    feats = [None]
    def feat_hook(m, i, o):
        feats[0].copy_(i.data.view(i.size(0), -1))
    model._modules.get("classifier").register_forward_hook(feat_hook)
    def feat_extr(x):
        feats[0] = torch.zeros(x.size(0), 2208)
        model(x)
        return feats[0]
    model.eval()
    return feat_extr