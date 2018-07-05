class Config(object):
    def __init__(self, name, classes, mean, std, path, shape, channel=3):
        self.name = name
        self.mean = mean
        self.std = std
        self.path = path
        self.classes = classes
        self.shape = shape
        self.channel = channel



cub = Config(
    'CUB-200-2011', 200, 
    [0.4856077, 0.49941534, 0.43237692],
    [0.23222743, 0.2277201, 0.26586822],
    './data',
    224,
)

