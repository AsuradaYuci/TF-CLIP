from __future__ import absolute_import

import math

import random
from torchvision.transforms import *
from PIL import Image
from torchvision.transforms import functional as F


class Random2DTranslation(object):
    """
    With a probability, first increase image size to (1 + 1/8), and then perform random crop.

    Args:
        height (int): target height.
        width (int): target width.
        p (float): probability of performing this transformation. Default: 0.5.
    """
    def __init__(self, height, width, p=0.5, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.p = p
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        if random.random() < self.p:
            return img.resize((self.width, self.height), self.interpolation)
        new_width, new_height = int(round(self.width * 1.125)), int(round(self.height * 1.125))
        resized_img = img.resize((new_width, new_height), self.interpolation)
        x_maxrange = new_width - self.width
        y_maxrange = new_height - self.height
        x1 = int(round(random.uniform(0, x_maxrange)))
        y1 = int(round(random.uniform(0, y_maxrange)))
        croped_img = resized_img.crop((x1, y1, x1 + self.width, y1 + self.height))
        return croped_img

class RandomErasing(object):
    """ Randomly selects a rectangle region in an image and erases its pixels.
        'Random Erasing Data Augmentation' by Zhong et al.
        See https://arxiv.org/pdf/1708.04896.pdf
    Args:
         probability: The probability that the Random Erasing operation will be performed.
         sl: Minimum proportion of erased area against input image.
         sh: Maximum proportion of erased area against input image.
         r1: Minimum aspect ratio of erased area.
         mean: Erasing value.
    """

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, img):

        if random.uniform(0, 1) >= self.probability:
            return img

        for attempt in range(100):
            area = img.size()[1] * img.size()[2]

            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < img.size()[2] and h < img.size()[1]:
                x1 = random.randint(0, img.size()[1] - h)
                y1 = random.randint(0, img.size()[2] - w)
                if img.size()[0] == 3:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                    img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                else:
                    img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                return img

        return img


class to_tensor(object):

    def __call__(self, imgs):
        tensor_imgs = []
        for img in imgs:
            tensor_imgs.append(F.to_tensor(img))

        return tensor_imgs

    def __repr__(self):
        return self.__class__.__name__ + '()'


class resize(Resize):

    def __init__(self, size=[128, 256], interpolation=Image.BILINEAR):
        super(resize, self).__init__(size, interpolation)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, imgs):
        resize_imgs = []
        for img in imgs:
            resize_imgs.append(F.resize(img, self.size, self.interpolation))
        return resize_imgs


class random_horizontal_flip(RandomHorizontalFlip):

    def __init__(self, p=0.5):
        super(random_horizontal_flip, self).__init__(p)

    def __call__(self, imgs):

        filp_imgs = []
        if random.random() < self.p:
            for img in imgs:
                filp_imgs.append(F.hflip(img))
            return filp_imgs
        else:
            return imgs


class pad(Pad):

    def __init__(self, padding, fill=0, padding_mode='constant'):
        super(pad, self).__init__(padding, fill, padding_mode)

    def __call__(self, imgs):
        pad_imgs = []
        for img in imgs:
            pad_imgs.append(F.pad(img, self.padding, self.fill, self.padding_mode))

        return pad_imgs


class random_crop(RandomCrop):

    def __init__(self, size, padding=0, pad_if_needed=False):
        super(random_crop, self).__init__(size, padding, pad_if_needed)

    def __call__(self, imgs):

        # print(len(imgs))
        i, j, h, w = self.get_params(imgs[0], self.size)

        crop_imgs = []
        for img in imgs:
            crop_imgs.append(F.crop(img, i, j, h, w))

        return crop_imgs


class normalize(Normalize):

    def __init__(self, mean, std):
        super(normalize, self).__init__(mean, std)

    def __call__(self, imgs):
        nor_imgs = []
        for img in imgs:
            nor_imgs.append(F.normalize(img, self.mean, self.std))
        return nor_imgs

class random_erasing(RandomErasing):

    def __init__(self, probability=0.5, sl=0.02, sh=0.4, r1=0.3, mean=(0.4914, 0.4822, 0.4465)):
        super(random_erasing, self).__init__(probability, sl, sh, r1, mean)

    def __call__(self, imgs):

        if random.uniform(0, 1) >= self.probability:
            return imgs

        C, H, W = imgs[0].size()

        for attempt in range(100):
            area = H * W
            target_area = random.uniform(self.sl, self.sh) * area
            aspect_ratio = random.uniform(self.r1, 1 / self.r1)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w < W and h < H:
                x1 = random.randint(0, H - h)
                y1 = random.randint(0, W - w)

                earse_imgs = []
                for img in imgs:
                    if C == 3:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                        img[1, x1:x1 + h, y1:y1 + w] = self.mean[1]
                        img[2, x1:x1 + h, y1:y1 + w] = self.mean[2]
                    else:
                        img[0, x1:x1 + h, y1:y1 + w] = self.mean[0]
                    earse_imgs.append(img)

                return earse_imgs

        return imgs

if __name__ == '__main__':
    pass
