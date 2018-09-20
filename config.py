class Data_config(object):
    def __init__(self, name, classes, mean, std, path, shape, channel=3):
        self.name = name
        self.mean = mean
        self.std = std
        if mean[0] > 1:
            self.mean = [a / 255 for a in mean]
            self.std = [a / 254 for a in std]
        self.path = path
        self.classes = classes
        self.shape = shape
        self.channel = channel



cub = Data_config(
    'CUB-200-2011', 200, 
    [0.4856077, 0.49941534, 0.43237692],
    [0.23222743, 0.2277201, 0.26586822],
    './data',
    224,
)

Al = Data_config(
    'Al', 12,
    [90.46230316, 158.09841919, 158.75947571],
    [60.16843796, 43.50852966, 45.99687958],
    './data',
    224,
)



