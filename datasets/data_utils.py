import random
import numpy as np

from PIL import Image
from PIL import ImageFilter

import torchvision.transforms.functional as TF
from torchvision import transforms
import torch


def to_tensor_and_norm(imgs, labels):
    # to tensor
    imgs = [TF.to_tensor(img) for img in imgs]
    labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
              for img in labels]

    imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            for img in imgs]
    return imgs, labels


def get_mean_std(data_name):
    if data_name == 'LEVIR':
        return [0.44647792, 0.4425377, 0.37772304], [0.21740167, 0.2035464, 0.18545838], [0.34149408, 0.33470634,
                                                                                          0.28560624], [0.15708335,
                                                                                                        0.15115038,
                                                                                                        0.1435679]
    elif data_name == 'CDD':
        return [0.35419008, 0.3916849, 0.34327868], [0.2165103, 0.23459111, 0.20905338],[0.4740057, 0.49984765, 0.46986604],[0.24348812, 0.2604329, 0.2572285]
    elif data_name == 'WHU':
        return [0.48525673, 0.44474944, 0.38744435], [0.17380719, 0.17025949, 0.17362548],[0.48447055, 0.48390314, 0.46201634],[0.21435988, 0.20399499, 0.21986482]
    elif data_name == 'SYSU':
        return [0.39638072, 0.5274554, 0.4634932],[0.2428711, 0.18390322, 0.18066421],[0.402662, 0.4859251, 0.3966497],[0.2172532, 0.18111998, 0.1786664]
    else:
        return [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]


class CDDataAugmentation:

    def __init__(
            self,
            img_size,
            with_random_hflip=False,
            with_random_vflip=False,
            with_random_rot=False,
            with_random_crop=False,
            with_scale_random_crop=False,
            with_random_blur=False,
            data_name=''
    ):
        self.img_size = img_size
        if self.img_size is None:
            self.img_size_dynamic = True
        else:
            self.img_size_dynamic = False
        self.with_random_hflip = with_random_hflip
        self.with_random_vflip = with_random_vflip
        self.with_random_rot = with_random_rot
        self.with_random_crop = with_random_crop
        self.with_scale_random_crop = with_scale_random_crop
        self.with_random_blur = with_random_blur

        self.mean1, self.std1, self.mean2, self.std2 = get_mean_std(data_name)

    def transform(self, imgs, labels, to_tensor=True):
        """
        :param imgs: [ndarray,]
        :param labels: [ndarray,]
        :return: [ndarray,],[ndarray,]
        """
        # resize image and covert to tensor
        imgs = [TF.to_pil_image(img) for img in imgs]
        if self.img_size is None:
            self.img_size = None

        if not self.img_size_dynamic:
            if imgs[0].size != (self.img_size, self.img_size):
                imgs = [TF.resize(img, [self.img_size, self.img_size], interpolation=3)
                        for img in imgs]
        else:
            self.img_size = imgs[0].size[0]

        labels = [TF.to_pil_image(img) for img in labels]
        if len(labels) != 0:
            if labels[0].size != (self.img_size, self.img_size):
                labels = [TF.resize(img, [self.img_size, self.img_size], interpolation=0)
                          for img in labels]

        random_base = 0.5
        if self.with_random_hflip and random.random() > 0.5:
            imgs = [TF.hflip(img) for img in imgs]
            labels = [TF.hflip(img) for img in labels]

        if self.with_random_vflip and random.random() > 0.5:
            imgs = [TF.vflip(img) for img in imgs]
            labels = [TF.vflip(img) for img in labels]

        if self.with_random_rot and random.random() > random_base:
            angles = [90, 180, 270]
            index = random.randint(0, 2)
            angle = angles[index]
            imgs = [TF.rotate(img, angle) for img in imgs]
            labels = [TF.rotate(img, angle) for img in labels]

        if self.with_random_crop and random.random() > 0:
            i, j, h, w = transforms.RandomResizedCrop(size=self.img_size). \
                get_params(img=imgs[0], scale=(0.8, 1.0), ratio=(1, 1))

            imgs = [TF.resized_crop(img, i, j, h, w,
                                    size=(self.img_size, self.img_size),
                                    interpolation=Image.CUBIC)
                    for img in imgs]

            labels = [TF.resized_crop(img, i, j, h, w,
                                      size=(self.img_size, self.img_size),
                                      interpolation=Image.NEAREST)
                      for img in labels]

        if self.with_scale_random_crop:
            # rescale
            scale_range = [1, 1.2]
            target_scale = scale_range[0] + random.random() * (scale_range[1] - scale_range[0])

            imgs = [pil_rescale(img, target_scale, order=3) for img in imgs]
            labels = [pil_rescale(img, target_scale, order=0) for img in labels]
            # crop
            imgsize = imgs[0].size  # h, w
            box = get_random_crop_box(imgsize=imgsize, cropsize=self.img_size)
            imgs = [pil_crop(img, box, cropsize=self.img_size, default_value=0)
                    for img in imgs]
            labels = [pil_crop(img, box, cropsize=self.img_size, default_value=255)
                      for img in labels]

        if self.with_random_blur and random.random() > 0:
            radius = random.random()
            imgs = [img.filter(ImageFilter.GaussianBlur(radius=radius))
                    for img in imgs]

        if to_tensor:
            # to tensor
            imgs = [TF.to_tensor(img) for img in imgs]
            labels = [torch.from_numpy(np.array(img, np.uint8)).unsqueeze(dim=0)
                      for img in labels]

            norm_imgs = [TF.normalize(imgs[0], mean=self.mean1, std=self.std1),
                         TF.normalize(imgs[1], mean=self.mean2, std=self.std2)]
            # imgs = [TF.normalize(img, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            #         for img in imgs]
        return norm_imgs, labels


def pil_crop(image, box, cropsize, default_value):
    assert isinstance(image, Image.Image)
    img = np.array(image)

    if len(img.shape) == 3:
        cont = np.ones((cropsize, cropsize, img.shape[2]), img.dtype) * default_value
    else:
        cont = np.ones((cropsize, cropsize), img.dtype) * default_value
    cont[box[0]:box[1], box[2]:box[3]] = img[box[4]:box[5], box[6]:box[7]]

    return Image.fromarray(cont)


def get_random_crop_box(imgsize, cropsize):
    h, w = imgsize
    ch = min(cropsize, h)
    cw = min(cropsize, w)

    w_space = w - cropsize
    h_space = h - cropsize

    if w_space > 0:
        cont_left = 0
        img_left = random.randrange(w_space + 1)
    else:
        cont_left = random.randrange(-w_space + 1)
        img_left = 0

    if h_space > 0:
        cont_top = 0
        img_top = random.randrange(h_space + 1)
    else:
        cont_top = random.randrange(-h_space + 1)
        img_top = 0

    return cont_top, cont_top + ch, cont_left, cont_left + cw, img_top, img_top + ch, img_left, img_left + cw


def pil_rescale(img, scale, order):
    assert isinstance(img, Image.Image)
    height, width = img.size
    target_size = (int(np.round(height * scale)), int(np.round(width * scale)))
    return pil_resize(img, target_size, order)


def pil_resize(img, size, order):
    assert isinstance(img, Image.Image)
    if size[0] == img.size[0] and size[1] == img.size[1]:
        return img
    if order == 3:
        resample = Image.BICUBIC
    elif order == 0:
        resample = Image.NEAREST
    return img.resize(size[::-1], resample)


def denormalize(tensor, data_name, type='A'):
    mean1, std1, mean2, std2 = get_mean_std(data_name)
    if type == 'A':
        mean = mean1
        std = std1
    if type == 'B':
        mean = mean2
        std = std2

    mean = torch.tensor(mean).reshape(1, 3, 1, 1)
    std = torch.tensor(std).reshape(1, 3, 1, 1)
    return tensor * std + mean
