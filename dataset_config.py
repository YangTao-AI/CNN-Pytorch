class DatasetConfig(object):
    def __init__(self, name, classes, mean, std, train_path, val_path, shape, channel=3):
        self.name = name
        self.mean = mean
        self.std = std
        if mean[0] > 1:
            self.mean = [a / 255 for a in mean]
            self.std = [a / 255 for a in std]
        self.train_path = train_path
        self.val_path = val_path
        self.classes = classes
        self.shape = shape
        self.channel = channel



cub = DatasetConfig(
    'CUB-200-2011', 200, 
    [0.4856077, 0.49941534, 0.43237692],
    [0.23222743, 0.2277201, 0.26586822],
    './data/train.zip',
    './data/val.zip',
    (224, 224),
)

Al = DatasetConfig(
    'Al', 12,
    [108.24723816, 169.19256592, 161.6643219],
    [65.68550873, 45.49485016, 48.18498611,],
    './data/train.zip',
    './data/val.zip',
    (224, 224),
)



