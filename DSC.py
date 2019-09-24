import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from irnn import irnn
from backbone.resnext.resnext101_regular import ResNeXt101

def conv1x1(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 1,
                    stride =stride, padding=0,bias=False)

def conv3x3(in_channels, out_channels, stride = 1):
    return nn.Conv2d(in_channels,out_channels,kernel_size = 3,
        stride =stride, padding=1,bias=False)

class Spacial_IRNN(nn.Module):
    def __init__(self,in_channels,alpha=1.0):
        super(Spacial_IRNN,self).__init__()
        self.left_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.right_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.up_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.down_weight = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,groups=in_channels,padding=0)
        self.left_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.up_weight.weight    = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.down_weight.weight  = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))

    def forward(self,input):
        return irnn()(input,self.up_weight.weight,self.right_weight.weight,self.down_weight.weight,self.left_weight.weight, self.up_weight.bias,self.right_weight.bias,self.down_weight.bias,self.left_weight.bias)

class Attention(nn.Module):
    def __init__(self,in_channels):
        super(Attention,self).__init__()
        self.out_channels = int(in_channels/2)
        self.conv1 = nn.Conv2d(in_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels,self.out_channels,kernel_size=3,padding=1,stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels,4,kernel_size=1,padding=0,stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self,x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out

class DSC_Module(nn.Module):
    def __init__(self,in_channels,out_channels,attention=1,alpha=1.0):
        super(DSC_Module,self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels,alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels,alpha)
        self.conv_in = conv1x1(in_channels,in_channels)
        self.conv2 = conv1x1(in_channels*4,in_channels)
        self.conv3 = conv1x1(in_channels*4,in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)
        
        
    
    def forward(self,x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up,top_right,top_down,top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv2(out)
        top_up,top_right,top_down,top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:,0:1,:,:])
            top_right.mul(weight[:,1:2,:,:])
            top_down.mul(weight[:,2:3,:,:])
            top_left.mul(weight[:,3:4,:,:])
        
        out = torch.cat([top_up,top_right,top_down,top_left],dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        
        return out

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x




class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y

class DSC(nn.Module):
    def __init__(self):
        super(DSC,self).__init__()

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4


        self.layer4_conv1 = LayerConv(2048, 512, 7, 1, 3, True)
        self.layer4_conv2 = LayerConv(512, 512, 7, 1, 3, True)
        self.layer4_dsc = DSC_Module(512, 512)
        self.layer4_conv3 = LayerConv(1024, 32, 1, 1, 0, False)

        self.layer3_conv1 = LayerConv(1024, 256, 5, 1, 2, True)
        self.layer3_conv2 = LayerConv(256, 256, 5, 1, 2, True)
        self.layer3_dsc = DSC_Module(256, 256)
        self.layer3_conv3 = LayerConv(512, 32, 1, 1, 0, False)

        self.layer2_conv1 = LayerConv(512, 128, 5, 1, 2, True)
        self.layer2_conv2 = LayerConv(128, 128, 5, 1, 2, True)
        self.layer2_dsc = DSC_Module(128, 128)
        self.layer2_conv3 = LayerConv(256, 32, 1, 1, 0, False)

        self.layer1_conv1 = LayerConv(256, 64, 3, 1, 1, True)
        self.layer1_conv2 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer1_dsc = DSC_Module(64, 64,alpha=0.8)
        self.layer1_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.layer0_conv1 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer0_conv2 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer0_dsc = DSC_Module(64, 64,alpha=0.8)
        self.layer0_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.relu = nn.ReLU()

        self.global_conv = LayerConv(160, 32, 1, 1, 0, True)

        self.layer4_predict = Predict(32, 1, 1)
        self.layer3_predict_ori = Predict(32, 1, 1)
        self.layer3_predict = Predict(2, 1, 1)
        self.layer2_predict_ori = Predict(32, 1, 1)
        self.layer2_predict = Predict(3, 1, 1)
        self.layer1_predict_ori = Predict(32, 1, 1)
        self.layer1_predict = Predict(4, 1, 1)
        self.layer0_predict_ori = Predict(32, 1, 1)
        self.layer0_predict = Predict(5, 1, 1)
        self.global_predict = Predict(32, 1, 1)
        self.fusion_predict = Predict(6, 1, 1)


    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_conv1 = self.layer4_conv1(layer4)
        layer4_conv2 = self.layer4_conv2(layer4_conv1)
        layer4_dsc = self.layer4_dsc(layer4_conv2)
        layer4_context = torch.cat((layer4_conv2, layer4_dsc), 1)
        layer4_conv3 = self.layer4_conv3(layer4_context)
        layer4_up = F.upsample(layer4_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer4_up = self.relu(layer4_up)

        layer3_conv1 = self.layer3_conv1(layer3)
        layer3_conv2 = self.layer3_conv2(layer3_conv1)
        layer3_dsc = self.layer3_dsc(layer3_conv2)
        layer3_context = torch.cat((layer3_conv2, layer3_dsc), 1)
        layer3_conv3 = self.layer3_conv3(layer3_context)
        layer3_up = F.upsample(layer3_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_up = self.relu(layer3_up)

        layer2_conv1 = self.layer2_conv1(layer2)
        layer2_conv2 = self.layer2_conv2(layer2_conv1)
        layer2_dsc = self.layer2_dsc(layer2_conv2)
        layer2_context = torch.cat((layer2_conv2, layer2_dsc), 1)
        layer2_conv3 = self.layer2_conv3(layer2_context)
        layer2_up = F.upsample(layer2_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_up = self.relu(layer2_up)

        layer1_conv1 = self.layer1_conv1(layer1)
        layer1_conv2 = self.layer1_conv2(layer1_conv1)
        layer1_dsc = self.layer1_dsc(layer1_conv2)
        layer1_context = torch.cat((layer1_conv2, layer1_dsc), 1)
        layer1_conv3 = self.layer1_conv3(layer1_context)
        layer1_up = F.upsample(layer1_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_up = self.relu(layer1_up)

        layer0_conv1 = self.layer0_conv1(layer0)
        layer0_conv2 = self.layer0_conv2(layer0_conv1)
        layer0_dsc = self.layer0_dsc(layer0_conv2)
        layer0_context = torch.cat((layer0_conv2, layer0_dsc), 1)
        layer0_conv3 = self.layer0_conv3(layer0_context)
        layer0_up = F.upsample(layer0_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer0_up = self.relu(layer0_up)

        global_concat = torch.cat((layer0_up, layer1_up, layer2_up, layer3_up, layer4_up), 1)
        global_conv = self.global_conv(global_concat)

        layer4_predict = self.layer4_predict(layer4_up)

        layer3_predict_ori = self.layer3_predict_ori(layer3_up)
        layer3_concat = torch.cat((layer3_predict_ori, layer4_predict), 1)
        layer3_predict = self.layer3_predict(layer3_concat)

        layer2_predict_ori = self.layer2_predict_ori(layer2_up)
        layer2_concat = torch.cat((layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer2_predict = self.layer2_predict(layer2_concat)

        layer1_predict_ori = self.layer1_predict_ori(layer1_up)
        layer1_concat = torch.cat((layer1_predict_ori, layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer1_predict = self.layer1_predict(layer1_concat)

        layer0_predict_ori = self.layer0_predict_ori(layer0_up)
        layer0_concat = torch.cat((layer0_predict_ori, layer1_predict_ori, layer2_predict_ori,
                                   layer3_predict_ori, layer4_predict), 1)
        layer0_predict = self.layer0_predict(layer0_concat)

        global_predict = self.global_predict(global_conv)

        # fusion
        fusion_concat = torch.cat((layer0_predict, layer1_predict, layer2_predict, layer3_predict,
                                   layer4_predict, global_predict), 1)
        fusion_predict = self.fusion_predict(fusion_concat)


        return layer4_predict, layer3_predict, layer2_predict, layer1_predict, layer0_predict, \
                   global_predict, fusion_predict
 
