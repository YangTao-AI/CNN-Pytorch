import torch.utils.data as data
import zipfile, pickle
from PIL import Image
from IPython import embed

import os

"""
    According to torchvision/dataserts/folder.py
"""

def has_file_allowed_extension(filename, extensions):
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def find_classes(zf):
    classes = [d.filename.split('/')[-2] for d in zf.filelist\
            if d.filename[-1] == '/' and len(d.filename.split('/')) == 3]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx

def make_dataset(zf, class_to_idx, extensions, dic):
    images = []
    for each in zf.filelist:
        fn = each.filename
        if dic and fn not in dic:
            continue
        if has_file_allowed_extension(fn, extensions):
            item = (fn, class_to_idx[fn.split('/')[-2]])
            images.append(item)
    return images

class ZipDatasetFolder(data.Dataset):
    def __init__(self, root, loader, extensions, transform=None,\
            target_transform=None, data_cached=False, use_cache=None,\
            num_workers=0, allow_dic=None):
        self.num_workers = num_workers
        print('[LOG] preparing data folder')
        
        self.loader = loader if loader else self.default_loader
        self.root = root
        self.extensions = extensions
        self.data_cached = data_cached
        if root[-4:] != '.zip':
            root += '.zip'
        self.zip = zipfile.ZipFile(root) 
        
        if use_cache and os.path.isfile(use_cache):
            with open(use_cache, 'rb') as f:
                classes, class_to_idx, samples = pickle.load(f)
        else:
            classes, class_to_idx = find_classes(self.zip)
            samples = make_dataset(self.zip, class_to_idx, extensions, dic=allow_dic)
            if use_cache:
                with open(use_cache, 'wb') as f:
                    pickle.dump([classes, class_to_idx, samples], f)
                    
        self.cache = [None for i in range(len(samples))]\
                if data_cached else None

        assert len(samples) > 0
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform
        print('[SUC] data folder')

    def __del__(self):
        self.zip.close()

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
        zf = zipfile.ZipFile(self.root)\
                if self.num_workers > 1 else self.zip
        with zf.open(path) as f:
            img = Image.open(f).convert('RGB')
        return img

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

class MyImageFolder(ZipDatasetFolder):
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

