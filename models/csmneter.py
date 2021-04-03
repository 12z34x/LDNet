import torch
import torch.nn as nn
# import sys
# import os.path as osp
# sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../'))
from .necks import ASPP, SELayer, BasicRFB
from .backbones import build_backbone
import torch.nn.functional as F
#***************************SANet**********************
# class BasicConv(nn.Module):
#     def __init__(self, in_channels, out_channels, use_bn=False, **kwargs):
#         super(BasicConv, self).__init__()
#         self.use_bn = use_bn
#         self.conv = nn.Conv2d(in_channels, out_channels, bias=not self.use_bn, **kwargs)
#         self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

#     def forward(self, x):
#         x = self.conv(x)
#         if self.use_bn:
#             x = self.bn(x)
#         return F.relu(x, inplace=True)
# class BasicDeconv(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size, stride=1, use_bn=False):
#         super(BasicDeconv, self).__init__()
#         self.use_bn = use_bn
#         self.tconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride, bias=not self.use_bn)
#         self.bn = nn.InstanceNorm2d(out_channels, affine=True) if self.use_bn else None

#     def forward(self, x):
#         # pdb.set_trace()
#         x = self.tconv(x)
#         if self.use_bn:
#             x = self.bn(x)
#         return F.relu(x, inplace=True)


# class SAModule_Head(nn.Module):
#     def __init__(self, in_channels, out_channels, use_bn):
#         super(SAModule_Head, self).__init__()
#         branch_out = out_channels // 4
#         self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
#                             kernel_size=1)
#         self.branch3x3 = BasicConv(in_channels, branch_out, use_bn=use_bn,
#                             kernel_size=3, padding=1)
#         self.branch5x5 = BasicConv(in_channels, branch_out, use_bn=use_bn,
#                             kernel_size=5, padding=2)
#         self.branch7x7 = BasicConv(in_channels, branch_out, use_bn=use_bn,
#                             kernel_size=7, padding=3)
    
#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#         branch3x3 = self.branch3x3(x)
#         branch5x5 = self.branch5x5(x)
#         branch7x7 = self.branch7x7(x)
#         out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
#         return out


# class SAModule(nn.Module):
#     def __init__(self, in_channels, out_channels, use_bn):
#         super(SAModule, self).__init__()
#         branch_out = out_channels // 4
#         self.branch1x1 = BasicConv(in_channels, branch_out, use_bn=use_bn,
#                             kernel_size=1)
#         self.branch3x3 = nn.Sequential(
#                         BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
#                             kernel_size=1),
#                         BasicConv(2*branch_out, branch_out, use_bn=use_bn,
#                             kernel_size=3, padding=1),
#                         )
#         self.branch5x5 = nn.Sequential(
#                         BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
#                             kernel_size=1),
#                         BasicConv(2*branch_out, branch_out, use_bn=use_bn,
#                             kernel_size=5, padding=2),
#                         )
#         self.branch7x7 = nn.Sequential(
#                         BasicConv(in_channels, 2*branch_out, use_bn=use_bn,
#                             kernel_size=1),
#                         BasicConv(2*branch_out, branch_out, use_bn=use_bn,
#                             kernel_size=7, padding=3),
#                         )
    
#     def forward(self, x):
#         branch1x1 = self.branch1x1(x)
#         branch3x3 = self.branch3x3(x)
#         branch5x5 = self.branch5x5(x)
#         branch7x7 = self.branch7x7(x)
#         out = torch.cat([branch1x1, branch3x3, branch5x5, branch7x7], 1)
#         return out
#*******************SANet*********************
class CSMNeter(nn.Module):
    def __init__(self, opt):
        super(CSMNeter, self).__init__()
        #***********SANet***************
        in_channels = 512
        use_bn=True
        # self.encoder = nn.Sequential(
        #     SAModule_Head(in_channels, 64, use_bn),
        #     #nn.MaxPool2d(2, 2),
        #     SAModule(64, 128, use_bn),
        #     #nn.MaxPool2d(2, 2),
        #     SAModule(128, 128, use_bn),
        #     #nn.MaxPool2d(2, 2),
        #     SAModule(128, 128, use_bn),
        #     )

        BatchNorm =nn.BatchNorm2d#!!!!
        self.AdaptiveAvgPool2d=torch.nn.AdaptiveAvgPool2d((1,1))
        self.tanh=torch.nn.Tanh()
        self.backbone = build_backbone(opt.backbone, opt.output_stride, BatchNorm)
        part_channels=64#64
        self.aspp = ASPP(opt.backbone,
                        opt.output_stride,
                        part_channels*2,#
                        BatchNorm)
        self.link_conv1 = nn.Sequential(nn.Conv2d(
        self.backbone.high_outc, part_channels*2, kernel_size=1, stride=1, padding=0, bias=False))

        self.linker=nn.Sequential(nn.Conv2d(part_channels, 128, kernel_size=3, stride=1, padding=1, bias=False))

        self.link_conv2 = nn.Sequential(nn.Conv2d(self.backbone.low_outc,part_channels*2, kernel_size=1, stride=1, padding=0, bias=False))#change in channels
        self.up2 = nn.Upsample(scale_factor=2)
        self.rfb = BasicRFB(part_channels*2, part_channels*2)
        self.region = nn.Sequential(nn.Conv2d(part_channels, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                     SELayer(128),
                                     nn.BatchNorm2d(128),#!!!!
                                     nn.ReLU(),
                                     nn.Conv2d(128, 2, kernel_size=1, stride=1))#two channels

        self.density = nn.Sequential(nn.Conv2d(part_channels*2, 128, kernel_size=3, stride=1, padding=1, bias=False),
                                     SELayer(128),
                                     nn.BatchNorm2d(128),#!!!!
                                     nn.ReLU(),
                                     nn.Conv2d(128, 1, kernel_size=1, stride=1))
        self._init_weight()
        if opt.freeze_bn:
            self.freeze_bn()

    def forward(self, input):

        mid_level_feat, low_level_feat = self.backbone(input)

        # attention=self.AdaptiveAvgPool2d(low_level_feat)
        # attention=self.tanh(attention[0][0])
        # attention=torch.clamp((attention+1)/2,min=0.9,max=0.95)
        #**********SANet**********
        #low_level_feat = self.encoder(low_level_feat)#
        low_level_feat = self.link_conv2(low_level_feat)
        mid_level_feat = self.link_conv1(mid_level_feat)
        #mid_level_feat = self.up2(mid_level_feat)
        x1=low_level_feat+mid_level_feat
        high1_level_feat = self.aspp(x1)#64 channels
        high2_level_feat= self.rfb(x1)#128 channels
        
 
        #x = self.up2(x)
        #x = F.interpolate(x, size=low_level_feat.shape[2:], mode='nearest') + low_level_feat

        # x = torch.cat((high1_level_feat,high2_level_feat), dim=1)

        #prediction of region, density
        # x=self.linker(high1_level_feat)+high2_level_feat
        region = self.region(high1_level_feat)
        density = self.density(high2_level_feat)
        return region, density

    def _init_weight(self):
        for module in [self.density, self.region]:
        # for module in [self.density]:
            for m in module.modules():
                if isinstance(m, nn.Conv2d):
                    torch.nn.init.kaiming_normal_(m.weight)
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.data.fill_(1)
                    m.bias.data.zero_()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                        or isinstance(m[1], nn.BatchNorm2d):
                    for p in m[1].parameters():
                        if p.requires_grad:
                            yield p
# #?
#     def get_10x_lr_params(self):
#         modules = [self.link_conv, self.density, self.region]
#         for i in range(len(modules)):
#             for m in modules[i].named_modules():
#                 if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
#                         or isinstance(m[1], nn.BatchNorm2d):
#                     for p in m[1].parameters():
#                         if p.requires_grad:
#                             yield p


if __name__ == "__main__":
    model = CSMNet(backbone='mobilenetv2', output_stride=16)
    model.eval()
    input = torch.rand(5, 3, 640, 480)
    output = model(input)
    pass
