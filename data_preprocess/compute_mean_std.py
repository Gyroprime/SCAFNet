import os
import numpy as np
from PIL import Image

if __name__ == '__main__':
    filepath = r"/home/jiawen/deep_learn/datasets/SYSU-CD-256/B/"  # Dataset directory
    pathDir = os.listdir(filepath)  # Images in dataset directory
    num = len(pathDir)  # Here (512512) is the size of each image
    print("Computing ...")
    data_mean = []
    data_std = []
    img_list = []
    for idx in range(len(pathDir)):
        filename = pathDir[idx]
        img = Image.open(os.path.join(filepath, filename))
        img = np.array(img)
        img = img[:, :, :, np.newaxis]
        img_list.append(img)
    imgs = np.concatenate(img_list, axis=3)
    imgs = imgs.astype(np.float32) / 255
    for i in range(3):
        pixels = imgs[:, :, i, :].ravel()
        data_mean.append(np.mean(pixels))
        data_std.append(np.std(pixels))

    print("mean:{}".format(data_mean))
    print("std:{}".format(data_std))
