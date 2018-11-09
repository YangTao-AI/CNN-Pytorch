import numpy as np
from IPython import embed
from main import Dataset
from dataset_config import *
from my_folder import *
from utils import *

dataset_cfg = cub
dataset = Dataset(dataset_cfg)
train_dataset = MyImageFolder(dataset.train_path)


pixels = 0
macro_mean, macro_mean2, micro_mean, micro_mean2 =\
        [np.zeros((3), dtype=np.float32) for i in range(4)]

n = len(train_dataset)
for i, [img, label] in enumerate(train_dataset):
    img = np.array(img, dtype=np.float32)

    macro_mean += img.mean(axis=0).mean(axis=0)
    macro_mean2 += (img**2).mean(axis=0).mean(axis=0)

    micro_mean += img.sum(axis=0).sum(axis=0)
    micro_mean2 += (img**2).sum(axis=0).sum(axis=0)

    pixels += img.shape[0] * img.shape[1]
    if (i + 1) % 100 == 0:
        cp.log('(#b)%.2f%%(##) (#y)%d(##)/(#y)%d(##)'%((i+1)/n*100, i+1, n))

macro_mean /= n
macro_mean2 /= n

micro_mean /= pixels
micro_mean2 /= pixels

micro_std = np.sqrt(micro_mean2 - (micro_mean ** 2))
macro_std = np.sqrt(macro_mean2 - (macro_mean ** 2))

cp('(#b)micro_mean(##): (#y){}(##), (#b)micro_std(##): (#y){}(##)'.format(
    list(micro_mean), list(micro_std)))
cp('(#b)macro_mean(##): (#y){}(##), (#b)macro_std(##): (#y){}(##)'.format(
    list(macro_mean), list(macro_std)))
