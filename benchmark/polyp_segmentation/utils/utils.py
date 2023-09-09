from .randaugment import SegCompose, SegResize, SegRandomRotate, SegRandomFlip, RandAugment, SegToTensor, SegNormalize
from torchvision import transforms


MEAN_STD = {
    'means': [
        [143.52408303, 83.56563752, 62.20275572],
        [102.20527125, 68.70382564, 46.94900982],
        [112.51202878, 73.19284487, 47.89626003],
        [153.3712074, 109.95507468, 94.92573678]
    ],
    'stds': [
        [80.36527342, 56.57798678, 48.3764685],
        [76.10290668, 52.26579002, 35.61231149],
        [79.33154067, 58.48307273, 42.46520215],
        [67.49728571, 60.77629461, 56.65900334]
    ]
}


class WeakStrongTransform(object):
    def __init__(self, mean, std):
        self.weak = SegCompose([
            SegResize(224),
            SegRandomRotate(180),
            SegRandomFlip()
        ])

        self.strong = transforms.Compose([
            RandAugment(n=3, m=10)])

        self.normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])

        self.tensor = transforms.Compose([
            transforms.ToTensor()])


def get_transform(dataidx):
    mean = [i / 255.0 for i in MEAN_STD['means'][dataidx]]
    std = [i / 255.0 for i in MEAN_STD['stds'][dataidx]]
    transform_ws = WeakStrongTransform(mean=mean, std=std)

    transform_train = SegCompose([
        SegResize(224),
        SegRandomRotate(180),
        SegRandomFlip(),
        SegToTensor(),
        SegNormalize(mean=mean, std=std)
    ])
    return transform_train, transform_ws
