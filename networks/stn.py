import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils
from IPython import embed
import torch.nn.functional as F
import numpy as np


class Stn(nn.Module):
    def __init__(self, cnn_localization, cnn, localization_fc, \
            theta=[1, 0, 0, 0, 1, 0]):
        super(Stn, self).__init__()

        # Spatial transformer localization-network
        self.localization = cnn_localization
        self.localization_fc = localization_fc
        self.cnn = cnn

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.localization_fc, 32),
            nn.ReLU(True),
            nn.Linear(32, 6)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(theta, \
                dtype=torch.float))

    def stn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.localization_fc)
        raw_theta = self.fc_loc(xs)
        theta = raw_theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, raw_theta

    def forward(self, x):
        stn_img, theta = self.stn(x)
        x = self.cnn(stn_img)
        return x, stn_img, theta

class Multi_stn(nn.Module):
    def __init__(self, num, model_func, num_classes, localization_fc=128,\
            pretrained=False, **kwargs):
        super(Multi_stn, self).__init__()
        theta = np.array([kwargs['theta'] for i in range(num)])
        theta = theta + np.random.uniform(-0.1, 0.1, size=(theta.shape))
        print(theta)
        self.stn = nn.ModuleList([model_func(num_classes, localization_fc,\
                pretrained, theta=theta[i]) for i in range(num)])

        self.ffc = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.7),
            nn.Linear(num_classes * num, num_classes)
        )
        self.num = num

    def forward(self, x):
        stns = [stn(x) for stn in self.stn]
        stn_imgs = torch.cat([stn[1].view(-1, 1, *stn[1].size()[1:])\
                for stn in stns], 1)
        stn_imgs = stn_imgs.view(-1, *stn_imgs.size()[2:])
        thetas = torch.cat([stn[2] for stn in stns], 0)
        x = torch.cat([stn[0] for stn in stns], 1)
        x = self.ffc(x)
        return x, stn_imgs, thetas

def multi_stn(*args, **kwargs):
    return Multi_stn(*args, **kwargs)

def stn_resnet18(num_classes, localization_fc=128,\
        pretrained=False, **kwargs):
    resnet18 = models.resnet18
    resnet18_local = resnet18(pretrained, num_classes=localization_fc)
    resnet18_cnn = resnet18(pretrained, num_classes=num_classes)
    return Stn(resnet18_local, resnet18_cnn, localization_fc, **kwargs)

def stn_resnext50(num_classes, localization_fc=128,\
        pretrained=False, **kwargs):
    from .resnext import resnext50
    resnext50_local = resnext50(pretrained, num_classes=localization_fc)
    resnext50_cnn = resnext50(pretrained, num_classes=num_classes)
    return Stn(resnext50_local, resnext50_cnn, localization_fc, **kwargs)

def stn_resnet50(num_classes, localization_fc=128,\
        pretrained=False, **kwargs):
    resnet50 = models.resnet50
    resnet50_local = resnet50(pretrained, num_classes=localization_fc)
    resnet50_cnn = resnet50(pretrained, num_classes=num_classes)
    return Stn(resnet50_local, resnet50_cnn, localization_fc, **kwargs)

if __name__ == '__main__':
    _stn_resnet18 = stn_resnet18(200, 128, True)
