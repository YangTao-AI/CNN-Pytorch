import torch.utils.data as data
import numpy as np
import zipfile, pickle
from PIL import Image
from IPython import embed
from utils import *
import os, csv
import torchvision

"""
    According to torchvision/dataserts/folder.py
"""

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

class DatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None,\
            target_transform=None, data_cached=False, use_cache=None,\
            num_workers=0, allow_dict=None):
        self.num_workers = num_workers
        with procedure('preparing data folder') as pp:
            self.loader = loader if loader else self.default_loader
            self.root = root
            self.extensions = extensions
            self.data_cached = data_cached

            _map = {}

            def make_hots(a):
                x = np.zeros((28), dtype=np.float32)
                x[a] = 1
                return x

            csv_path = root + '.csv'
            if os.path.isfile(csv_path):
                with open(csv_path, newline='') as csvfile:
                    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
                    for row in spamreader:
                        if row[0] == 'Id':
                            continue
                        _map[row[0]] = make_hots(list(map(int, row[1].split(' '))))
            else:
                files = os.listdir(root)
                _map = {_f.split('_')[0]: _f.split('_')[0] for _f in files if _f.split('.')[-1] == 'png'}



            num_classses = 28
            classes = list(range(num_classses))


            if isinstance(allow_dict, list):
                allow_dict = set(allow_dict)

            if allow_dict is not None:
                self.samples = [[item, _map[item]] for item in _map.keys() if item in allow_dict]
            else:
                self.samples = [[item, _map[item]] for item in _map.keys()]
                        
            self.cache = [None for i in range(len(self.samples))]\
                    if data_cached else None

            self.classes = classes

            self.transform = transform
            self.target_transform = target_transform
            pp.msg += ' (#g){}(#)'.format(self.__len__())

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        if self.data_cached and self.cache[index]:
            sample, target = self.cache[index]
            if self.transform is not None:
                sample = self.transform(sample)
            if self.target_transform is not None:
                target = self.target_transform(target)
            return sample, target

        path, target = self.samples[index]
        sample = self.loader(path)
        if self.data_cached:
            self.cache[index] = [sample, target]

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def default_loader(self, path):
        imgs = []
        for suffix in ['green', 'blue', 'red', 'yellow']:
            _path = os.path.join(self.root, '{}_{}.png'.format(path, suffix))
            with open(_path, 'rb') as f:
                img = Image.open(f).convert('RGB')
                imgs.append(np.array(img))
        imgs = np.concatenate([img[:,:,:1] for img in imgs], axis=2)
        return Image.fromarray(imgs)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class MyImageFolder(DatasetFolder):
    """
    Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """


    def __init__(self, root, transform=None, target_transform=None,\
            loader=None, **kwargs):

        super(MyImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,\
                transform=transform, target_transform=target_transform,\
                **kwargs)

        self.imgs = self.samples

