import os
import numpy as np
from PIL import Image

dataidxs = 0
path = '/data/zbm/QaTa-COV19-v2/'
datamap = {
    0: 'Train Set',
    1: 'Test Set'
}

sum = np.empty((0, 3), np.uint8)

for dataidxs in range(2):
    img_path = os.path.join(path, datamap[dataidxs], 'Images')
    mask_path = os.path.join(path, datamap[dataidxs], 'Ground-truths')
    img_list = sorted(os.listdir(img_path))
    mask_list = sorted(os.listdir(mask_path))
    img_list = [os.path.join(img_path, i) for i in img_list]
    mask_list = [os.path.join(mask_path, i) for i in mask_list]

    # assert len(img_list) == len(mask_list)
    # index = range(len(img_list))
    # index_test = [i % 10 == 8 or i % 10 == 9 for i in index]
    # index_train = [i is False for i in index_test]
    # img_list = np.array(img_list)
    # mask_list = np.array(mask_list)

    # sum = np.empty((0, 3), np.uint8)
    for img_path in img_list:
        img = Image.open(img_path).convert("RGB")
        img = np.array(img)
        img = img.reshape((-1, 3))
        sum = np.concatenate((sum, img), axis=0)

mean = np.mean(sum, axis=0)
std = np.std(sum, axis=0)

# print(dataidxs)
print(mean)
print(std)
