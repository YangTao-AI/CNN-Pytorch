import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils
from IPython import embed
import torch.nn.functional as F
import numpy as np


class ctn(nn.Module):
    def __init__(self, cnn_localization, cnn, localization_fc, \
            theta=[0, 0]):
        if len(theta) == 6:
            theta = theta[2::3]
        print(theta)
        super(ctn, self).__init__()

        # Spatial transformer localization-network
        self.localization = cnn_localization
        self.localization_fc = localization_fc
        self.cnn = cnn
        self.size = 0.5

        # Regressor for the 3 * 2 affine matrix
        self.fc_loc = nn.Sequential(
            nn.Linear(self.localization_fc, 32),
            nn.ReLU(True),
            nn.Linear(32, 2)
        )

        # Initialize the weights/bias with identity transformation
        self.fc_loc[2].weight.data.zero_()
        self.fc_loc[2].bias.data.copy_(torch.tensor(theta, \
                dtype=torch.float))

        

    def ctn(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, self.localization_fc)
        raw_theta = self.fc_loc(xs)
        
        s = (torch.eye(2)*0.5).unsqueeze(0).repeat(x.size()[0], 1, 1).cuda()
        theta = torch.cat([s, raw_theta.view(-1, 2, 1)], 2)

        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x, raw_theta

    def forward(self, x):
        ctn_img, theta = self.ctn(x)
        x = self.cnn(ctn_img)
        return x, ctn_img, theta

class Multi_ctn(nn.Module):
    def __init__(self, num, model_func, num_classes, localization_fc=128,\
            pretrained=False, **kwargs):
        super(Multi_ctn, self).__init__()
        theta = np.array([kwargs['theta'] for i in range(num)])
        theta = theta + np.random.uniform(-0.5, 0.5, size=(theta.shape))
        self.stn = nn.ModuleList([model_func(num_classes, localization_fc,\
                pretrained, theta=theta[i]) for i in range(num)])

        self.ffc = nn.Sequential(
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(num_classes * num, num_classes)
        )
        self.num = num

    def forward(self, x):
        ctns = [ctn(x) for ctn in self.stn]
        ctn_imgs = torch.cat([ctn[1].view(-1, 1, *ctn[1].size()[1:])\
                for ctn in ctns], 1)
        ctn_imgs = ctn_imgs.view(-1, *ctn_imgs.size()[2:])
        thetas = torch.cat([ctn[2] for ctn in ctns], 0)
        x = torch.cat([ctn[0] for ctn in ctns], 1)
        x = self.ffc(x)
        return x, ctn_imgs, thetas

def multi_ctn(*args, **kwargs):
    return Multi_ctn(*args, **kwargs)

def ctn_resnet18(num_classes, localization_fc=128,\
        pretrained=False, **kwargs):
    resnet18 = models.resnet18
    resnet18_local = resnet18(pretrained, num_classes=localization_fc)
    resnet18_cnn = resnet18(pretrained, num_classes=num_classes)
    return ctn(resnet18_local, resnet18_cnn, localization_fc, **kwargs)

def ctn_resnet50(num_classes, localization_fc=128,\
        pretrained=False, **kwargs):
    resnet50 = models.resnet50
    resnet50_local = resnet50(pretrained, num_classes=localization_fc)
    resnet50_cnn = resnet50(pretrained, num_classes=num_classes)
    return ctn(resnet50_local, resnet50_cnn, localization_fc, **kwargs)

if __name__ == '__main__':
    _ctn_resnet18 = ctn_resnet18(200, 128, True)
