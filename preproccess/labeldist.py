import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


train_path = '/data/zbm/QaTa-COV19-v2/Train Set/Ground-truths'

file_list = os.listdir(train_path)

pars = []

for f in file_list:
    img = Image.open(train_path + '/' + f).convert('L')
    img = np.array(img)
    sum_pix = img.shape[0] * img.shape[1]
    img[img < 100] = 0
    img[img > 100] = 1
    sum_mask = np.sum(img)
    pars.append(sum_mask / sum_pix)

# x = np.arange(0, 1, 0.1)
y = pars

plt.hist(y, bins=20)
plt.show()
