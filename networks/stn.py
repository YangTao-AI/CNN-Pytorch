import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.utils
from IPython import embed
import torch.nn.functional as F


class Stn(nn.Module):
    def __init__(self, cnn_localization, cnn, localization_fc):
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
        self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

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

def stn_resnet18(num_classes, localization_fc=128, pretrained=False):
    resnet18 = models.resnet18
    resnet18_local = resnet18(pretrained, num_classes=localization_fc)
    resnet18_cnn = resnet18(pretrained, num_classes=num_classes)
    return Stn(resnet18_local, resnet18_cnn, localization_fc)

if __name__ == '__main__':
    _stn_resnet18 = stn_resnet18(200, 128, True)
