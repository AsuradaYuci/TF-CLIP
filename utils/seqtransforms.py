import torch
import math
import random
from PIL import Image, ImageOps
import numpy as np


class Compose(object):
    """Composes several transforms together.

    Args:
        transforms (List[Transform]): list of transforms to compose.

    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, seqs):
        for t in self.transforms:
            seqs = t(seqs)
        return seqs


class RectScale(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, seqs):  # seqs = list[[image0,...,image8]]
        modallen = len(seqs)  # 1个list
        framelen = len(seqs[0])  # 8
        new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]  # <class 'list'>: [[[], [], [], [], [], [], [], []]]

        for modal_ind, modal in enumerate(seqs):  # 遍历modal，总共有1个
            for frame_ind, frame in enumerate(modal):  # 遍历每一帧图片
                w, h = frame.size  # w：128，h：256
                if h == self.height and w == self.width:
                    new_seqs[modal_ind][frame_ind] = frame
                else:
                    new_seqs[modal_ind][frame_ind] = frame.resize((self.width, self.height), self.interpolation)

        return new_seqs


class RandomSizedRectCrop(object):
    def __init__(self, height, width, interpolation=Image.BILINEAR):
        self.height = height
        self.width = width
        self.interpolation = interpolation

    def __call__(self, seqs):
        sample_img = seqs[0][0]
        for attempt in range(10):
            area = sample_img.size[0] * sample_img.size[1]
            target_area = random.uniform(0.64, 1.0) * area
            aspect_ratio = random.uniform(2, 3)

            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))

            if w <= sample_img.size[0] and h <= sample_img.size[1]:
                x1 = random.randint(0, sample_img.size[0] - w)
                y1 = random.randint(0, sample_img.size[1] - h)

                sample_img = sample_img.crop((x1, y1, x1 + w, y1 + h))
                assert (sample_img.size == (w, h))
                modallen = len(seqs)
                framelen = len(seqs[0])
                new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]

                for modal_ind, modal in enumerate(seqs):
                    for frame_ind, frame in enumerate(modal):

                        frame = frame.crop((x1, y1, x1 + w, y1 + h))
                        new_seqs[modal_ind][frame_ind] = frame.resize((self.width, self.height), self.interpolation)

                return new_seqs

        # Fallback
        scale = RectScale(self.height, self.width,
                          interpolation=self.interpolation)
        return scale(seqs)


class RandomSizedEarser(object):

    def __init__(self, sl=0.02, sh=0.2, asratio=0.3, p=0.5):
        self.sl = sl
        self.sh = sh
        self.asratio = asratio
        self.p = p

    def __call__(self, seqs):
        modallen = len(seqs)
        framelen = len(seqs[0])
        new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]
        for modal_ind, modal in enumerate(seqs):
            for frame_ind, frame in enumerate(modal):
                p1 = random.uniform(0.0, 1.0)
                W = frame.size[0]
                H = frame.size[1]
                area = H * W

                if p1 > self.p:
                    new_seqs[modal_ind][frame_ind] = frame
                else:
                    gen = True
                    while gen:
                        Se = random.uniform(self.sl, self.sh) * area
                        re = random.uniform(self.asratio, 1 / self.asratio)
                        He = np.sqrt(Se * re)
                        We = np.sqrt(Se / re)
                        xe = random.uniform(0, W - We)
                        ye = random.uniform(0, H - He)
                        if xe + We <= W and ye + He <= H and xe > 0 and ye > 0:
                            x1 = int(np.ceil(xe))
                            y1 = int(np.ceil(ye))
                            x2 = int(np.floor(x1 + We))
                            y2 = int(np.floor(y1 + He))
                            part1 = frame.crop((x1, y1, x2, y2))
                            Rc = random.randint(0, 255)
                            Gc = random.randint(0, 255)
                            Bc = random.randint(0, 255)
                            I = Image.new('RGB', part1.size, (Rc, Gc, Bc))
                            frame.paste(I, part1.size)
                            break

                    new_seqs[modal_ind][frame_ind] = frame

        return new_seqs


class RandomHorizontalFlip(object):
    """Randomly horizontally flips the given PIL.Image Sequence with a probability of 0.5
        """
    def __call__(self, seqs):
        if random.random() < 0.5:
            modallen = len(seqs)
            framelen = len(seqs[0])
            new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]
            for modal_ind, modal in enumerate(seqs):
                for frame_ind, frame in enumerate(modal):
                    new_seqs[modal_ind][frame_ind] = frame.transpose(Image.FLIP_LEFT_RIGHT)
            return new_seqs
        return seqs


class ToTensor(object):

    def __call__(self, seqs):
        modallen = len(seqs)
        framelen = len(seqs[0])
        new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]
        pic = seqs[0][0]

        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)

        if pic.mode == 'I':
            for modal_ind, modal in enumerate(seqs):
                for frame_ind, frame in enumerate(modal):
                    img = torch.from_numpy(np.array(frame, np.int32, copy=False))
                    img = img.view(pic.size[1], pic.size[0], nchannel)
                    new_seqs[modal_ind][frame_ind] = img.transpose(0, 1).transpose(0, 2).contiguous()

        elif pic.mode == 'I;16':
            for modal_ind, modal in enumerate(seqs):
                for frame_ind, frame in enumerate(modal):
                    img = torch.from_numpy(np.array(frame, np.int16, copy=False))
                    img = img.view(pic.size[1], pic.size[0], nchannel)
                    new_seqs[modal_ind][frame_ind] = img.transpose(0, 1).transpose(0, 2).contiguous()
        else:
            for modal_ind, modal in enumerate(seqs):
                for frame_ind, frame in enumerate(modal):
                    img = torch.ByteTensor(torch.ByteStorage.from_buffer(frame.tobytes()))
                    img = img.view(pic.size[1], pic.size[0], nchannel)
                    img = img.transpose(0, 1).transpose(0, 2).contiguous()
                    new_seqs[modal_ind][frame_ind] = img.float().div(255)

        return new_seqs


class Normalize(object):
    """Given mean: (R, G, B) and std: (R, G, B),
        will normalize each channel of the torch.*Tensor, i.e.
        channel = (channel - mean) / std
    """
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, seqs):
        # TODO: make efficient
        modallen = len(seqs)
        framelen = len(seqs[0])
        new_seqs = [[[] for _ in range(framelen)] for _ in range(modallen)]

        for modal_ind, modal in enumerate(seqs):
            for frame_ind, frame in enumerate(modal):
                for t, m, s in zip(frame, self.mean, self.std):
                    t.sub_(m).div_(s)
                    new_seqs[modal_ind][frame_ind] = frame

        return new_seqs
