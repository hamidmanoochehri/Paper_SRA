# original code obtained from https://github.com/facebookresearch/moco-v3
# ------------------------------------------------------------------------------
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from PIL import Image, ImageFilter, ImageOps
import math
import random
import torchvision.transforms.functional as tf

class TransformWrapper:
    def __init__(self, transform, use_adaptive_params=False):
        self.transform = transform
        self.use_adaptive_params = use_adaptive_params

    def __call__(self, x, adaptive_params=None):
        if self.use_adaptive_params and adaptive_params is not None:
            return self.transform(x, adaptive_params)
        else:
            return self.transform(x)


class CustomCompose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x, adaptive_params=None):
        for t in self.transforms:
            x = t(x, adaptive_params)
        return x


class TwoCropsTransform:
    """Take two random crops of one image, pass adaptive_params if required"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x, adaptive_params=None):
        #print(f"TwoCropsTransform __call__: Received adaptive_params: {adaptive_params}")
        #if not adaptive_params:
        #    print(f'*** WARNING *** TwoCropsTransform.__call__ fn adaptive_params is FALSE. rgb_he_wrgb augmentation might not work properly')
        im1 = self.base_transform1(x, adaptive_params) if adaptive_params else self.base_transform1(x)
        im2 = self.base_transform2(x, adaptive_params) if adaptive_params else self.base_transform2(x)
        return [im1, im2]


class GaussianBlur(object):
    """Gaussian blur augmentation from SimCLR: https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)
