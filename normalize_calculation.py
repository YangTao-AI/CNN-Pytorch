import cv2
import numpy as np
from IPython import embed

from my_folder import *


train  = MyImageFolder('../data_2/train', use_cache='../data_2/train.pkl')
ae, ae2, ie, ie2 = [np.zeros((3), dtype=np.float32) for i in range(4)]
num = 0

for each in train:
    a = np.array(each[0], dtype=np.float32)/255.0
    ae += a.mean(axis=0).mean(axis=0)
    ae2 += (a**2).mean(axis=0).mean(axis=0)


    ie += a.sum(axis=0).sum(axis=0)
    ie2 += (a**2).sum(axis=0).sum(axis=0)
    num += a.shape[0] * a.shape[1]

ae /= len(train)
ae2 /= len(train)

ie /= num
ie2 /= num
print('Macro average:', ae, 'std:', np.sqrt(np.abs(ae**2 - ae2)))
print('Micro average:', ie, 'std:', np.sqrt(np.abs(ie**2 - ie2)))
