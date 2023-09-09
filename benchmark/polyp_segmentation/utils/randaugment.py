import random
import numbers
import PIL

import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as F

# ---------------------------override---------------------------


class SegCompose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, target):
        for t in self.transforms:
            img, target = t(img, target)
        return img, target


class SegToTensor(object):
    def __init__(self):
        self.tensor = transforms.ToTensor()

    def __call__(self, img, target):
        return self.tensor(img), self.tensor(target)
        # return self.tensor(img), torch.tensor(np.array(target))


class SegNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor, target):
        return F.normalize(tensor, self.mean, self.std), target


class SegResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        else:
            self.size = size

    def __call__(self, img, target):
        return F.resize(img, self.size), F.resize(target, self.size)


class SegRandomFlip(object):
    def __call__(self, img, target):
        flip_mode = random.randint(-1, 2)
        if flip_mode == 2:
            return img, target
        elif flip_mode == 1:
            return F.hflip(img), F.hflip(target)
        else:
            return F.vflip(img), F.vflip(target)


class SegRandomRotate(object):
    def __init__(self, degrees):
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            if len(degrees) != 2:
                raise ValueError("If degrees is a sequence, it must be of len 2.")
            self.degrees = degrees

    def __call__(self, img, target):
        angle = random.uniform(self.degrees[0], self.degrees[1])
        return img.rotate(angle), target.rotate(angle)


# ---------------------------override---------------------------

PARAMETER_MAX = 10


def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img)


def Brightness(img, **kwarg):
    v = random.uniform(0, 2)
    return PIL.ImageEnhance.Brightness(img).enhance(v)


def Contrast(img, **kwarg):
    v = random.uniform(0, 2)
    return PIL.ImageEnhance.Contrast(img).enhance(v)


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img)


def Identity(img, **kwarg):
    return img


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v)


def Sharpness(img, **kwargs):
    v = random.uniform(0, 3)
    return PIL.ImageEnhance.Sharpness(img).enhance(v)


def RandomBlur(img, **kwargs):
    v = random.randint(0, 3)
    if v == 1:
        r = random.uniform(1, 10)
        return img.filter(PIL.ImageFilter.GaussianBlur(r))
    elif v == 2:
        r = random.uniform(1, 10)
        return img.filter(PIL.ImageFilter.BoxBlur(r))
    else:
        return img


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def gaussuanBlur(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return img.filter(PIL.ImageFilter.GaussianBlur(v))


def boxBlur(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return img.filter(PIL.ImageFilter.BoxBlur(v))

# ---------------------------randomAug---------------------------


class RandAugment(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = self.__get_augment_pool__()

    def __get_augment_pool__(self):
        augs = [
            (AutoContrast, None, None),  # 最大图像对比度，计算一个输入图像的直方图，从这个直方图中去除最亮和最暗的百分之cutoff，然后重新映射图像，以便保留的最暗像素变为黑色，即0，最亮的变为白色，即255。
            (Brightness, 0.95, 0.05),     # 增加图片亮度
            # (Color, 0.95, 0.05),          # 图片色度增强
            (Contrast, 0.95, 0.05),       # 图片对比度增强
            (Equalize, None, None),      # 均衡图像直方图。此功能将非线性映射应用于输入图像，以便在输出图像中创建灰度值得均匀分布
            # (Identity, None, None),      # 啥也没做
            (Posterize, 4, 4),           # 将每个颜色通道上变量bits对应的低(8-bits)个bit置0。变量bits的取值范围为[0，8]。
            (Sharpness, 0.95, 0.05),      # 图片锐度增强
            (RandomBlur, None, None),
            (gaussuanBlur, 5, 0.05),
            (boxBlur, 5, 0.05),
        ]
        return augs

    def __call__(self, img):
        ops = random.sample(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            v = np.random.randint(1, self.m)
            if random.random() < 0.5:
                img = op(img, v=v, max_v=max_v, bias=bias)
        return img
