from .randaugment import SegCompose, SegResize, SegRandomRotate, SegRandomFlip, RandAugment, SegToTensor, SegNormalize
from torchvision import transforms


MEAN_STD = {
    'means': [138.65198969, 138.65198969, 138.65198969],
    'stds': [60.06416719, 60.06416719, 60.06416719]
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


def get_transform():
    mean = [i / 255.0 for i in MEAN_STD['means']]
    std = [i / 255.0 for i in MEAN_STD['stds']]
    transform_ws = WeakStrongTransform(mean=mean, std=std)

    transform_train = SegCompose([
        SegResize(224),
        SegRandomRotate(180),
        SegRandomFlip(),
        SegToTensor(),
        SegNormalize(mean=mean, std=std)
    ])

    transform_test = SegCompose([
        SegResize(224),
        SegToTensor(),
        SegNormalize(mean=mean, std=std)
    ])
    return transform_train, transform_ws, transform_test
