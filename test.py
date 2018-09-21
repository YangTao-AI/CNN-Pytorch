from main import main
from IPython import embed

from my_folder import MyImageFolder

import torchvision.transforms as transforms
import torch
from config import Al as dataset
import numpy as np

normalize = transforms.Normalize(dataset.mean, dataset.std)
data = MyImageFolder(
    './data/test.zip', 
    transforms.Compose([
        transforms.Resize(dataset.shape),
        transforms.CenterCrop(dataset.shape),
        transforms.ToTensor(),
        normalize,
    ])
)

model = main(get_model=True).cuda()

f = open('./resnet50.txt', 'w')
batch_size = 64
for i in range(0, len(data), batch_size):
    k = min(i + batch_size, len(data))
    _input = torch.cat([
        data[j][0].reshape(
            1, *data[j][0].shape
        ) for j in range(i, k)
    ])
    # print(_input.shape)
    x = _input.cuda()
    y = model(x)
    z = np.array(y.argmax(dim=1))
    
    for j in range(z.shape[0]):
        s = '%s %s\n' % (
                data.samples[i+j][0], z[j])
        print(s)
        f.write(s)

f.close()
embed()
