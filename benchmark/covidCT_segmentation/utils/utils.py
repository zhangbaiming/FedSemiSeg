from .randaugment import SegCompose, SegResize, SegRandomRotate, SegRandomFlip, RandAugment, SegToTensor, SegNormalize
from torchvision import transforms


MEAN_STD = {
    'means': [
        [135.90649616, 135.90649616, 135.90649616],
        [39.62653738, 39.62653738, 39.62653738],
        [119.57933762, 119.57933762, 119.57933762],
        [90.00803375, 90.00803375, 90.00803375]
    ],
    'stds': [
        [84.57924375, 84.57924375, 84.57924375],
        [42.98936399, 42.98936399, 42.98936399],
        [89.61193969, 89.61193969, 89.61193969],
        [56.71577551, 56.71577551, 56.71577551]
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
