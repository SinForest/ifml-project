import torchvision.models as tvm
import torch.nn as nn
import torch


def alex_fc6(cuda):
    model = tvm.alexnet(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:3])
    model.classifier = nc
    model.eval()
    if cuda: model.cuda()
    return model

def alex_fc7(cuda):
    model = tvm.alexnet(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:-1])
    model.classifier = nc
    model.eval()
    if cuda: model.cuda()
    return model

def vgg19bn_fc6(cuda):
    model = tvm.vgg19_bn(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:2])
    model.classifier = nc
    model.eval()
    if cuda: model.cuda()
    return model

def vgg19bn_fc7(cuda):
    model = tvm.vgg19_bn(pretrained=True)
    nc = nn.Sequential(*list(model.classifier.children())[:-2])
    model.classifier = nc
    model.eval()
    if cuda: model.cuda()
    return model

def res50_avg(cuda):
    model = tvm.resnet50(pretrained=True)
    feats = [None]
    def feat_hook(m, i, o):
        feats[0].copy_(o.data[:, :, 0, 0])
    model._modules.get("avgpool").register_forward_hook(feat_hook)
    def feat_extr(x):
        feats[0] = torch.zeros(x.size(0), 2048)
        model(x)
        return torch.autograd.Variable(feats[0])
    model.eval()
    if cuda: model.cuda()
    return feat_extr

def dense161_last(cuda):
    model = tvm.densenet161(pretrained=True)
    feats = [None]
    def feat_hook(m, i, o):
        feats[0].copy_(i[0].data.view(i[0].size(0), -1))
    model._modules.get("classifier").register_forward_hook(feat_hook)
    def feat_extr(x):
        feats[0] = torch.zeros(x.size(0), 2208)
        model(x)
        return torch.autograd.Variable(feats[0])
    model.eval()
    if cuda: model.cuda()
    return feat_extr