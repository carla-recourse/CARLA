from __future__ import division

import numpy as np
import torch
from numpy.random import binomial, uniform
from PIL import Image
from torchvision import transforms


class top_masker:
    """
    Returned mask is sampled from component-wise independent Bernoulli
    distribution with probability of component to be unobserved p.
    Such mask induces the type of missingness which is called
    in literature "missing completely at random" (MCAR).
    If some value in batch is missed, it automatically becomes unobserved.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        pp = uniform(low=0.0, high=self.p, size=batch.shape[0])
        pp = np.expand_dims(pp, axis=1)
        pp = np.repeat(pp, batch.shape[1], axis=1)
        nan_mask = torch.isnan(batch).float()  # missed values
        #         bernoulli_mask_numpy = np.random.choice(2, size=batch.shape,
        #                                                 p=[1 - pp, pp])
        bernoulli_mask_numpy = binomial(1, pp, size=None)
        #         print(bernoulli_mask_numpy.shape)
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        mask = torch.max(bernoulli_mask, nan_mask)  # logical or
        return mask


class RandomPattern:
    """
    Reproduces "random pattern mask" for inpainting, which was proposed in
    Pathak, D., Krahenbuhl, P., Donahue, J., Darrell, T.,
    & Efros, A. A. Context Encoders: Feature Learning by Inpainting.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1604.07379
    This code is based on lines 273-283 and 316-330 of Context Encoders
    implementation:
    https://github.com/pathak22/context-encoder/blob/master/train_random.lua
    The idea is to generate small matrix with uniform random elements,
    then resize it using bicubic interpolation into a larger matrix,
    then binarize it with some threshold,
    and then crop a rectangle from random position and return it as a mask.
    If the rectangle contains too many or too few ones, the position of
    the rectangle is generated again.
    The big matrix is resampled when the total number of elements in
    the returned masks times update_freq is more than the number of elements
    in the big mask. This is done in order to find balance between generating
    the big matrix for each mask (which is involves a lot of unnecessary
    computations) and generating one big matrix at the start of the training
    process and then sampling masks from it only (which may lead to
    overfitting to the specific patterns).
    """

    def __init__(
        self, max_size=10000, resolution=0.06, density=0.25, update_freq=1, seed=239
    ):
        """
        Args:
            max_size (int):      the size of big binary matrix
            resolution (float):  the ratio of the small matrix size to
                                 the big one. Authors recommend to use values
                                 from 0.01 to 0.1.
            density (float):     the binarization threshold, also equals
                                 the average ones ratio in the mask
            update_freq (float): the frequency of the big matrix resampling
            seed (int):          random seed
        """
        self.max_size = max_size
        self.resolution = resolution
        self.density = density
        self.update_freq = update_freq
        self.rng = np.random.RandomState(seed)
        self.regenerate_cache()

    def regenerate_cache(self):
        """
        Resamples the big matrix and resets the counter of the total
        number of elements in the returned masks.
        """
        low_size = int(self.resolution * self.max_size)
        low_pattern = self.rng.uniform(0, 1, size=(low_size, low_size)) * 255
        low_pattern = torch.from_numpy(low_pattern.astype("float32"))
        pattern = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(self.max_size, Image.BICUBIC),
                transforms.ToTensor(),
            ]
        )(low_pattern[None])[0]
        pattern = torch.lt(pattern, self.density).byte()
        self.pattern = pattern.byte()
        self.points_used = 0

    def __call__(self, batch, density_std=0.05):
        """
        Batch is supposed to have shape [num_objects x num_channels x
        x width x height].
        Return binary mask of the same shape, where for each object
        the ratio of ones in the mask is in the open interval
        (self.density - density_std, self.density + density_std).
        The less is density_std, the longer is mask generation time.
        For very small density_std it may be even infinity, because
        there is no rectangle in the big matrix which fulfills
        the requirements.
        """
        batch_size, num_channels, width, height = batch.shape
        res = torch.zeros_like(batch, device="cpu")
        idx = list(range(batch_size))
        while idx:
            nw_idx = []
            x = self.rng.randint(0, self.max_size - width + 1, size=len(idx))
            y = self.rng.randint(0, self.max_size - height + 1, size=len(idx))
            for i, lx, ly in zip(idx, x, y):
                res[i] = self.pattern[lx : lx + width, ly : ly + height][None]
                coverage = float(res[i, 0].mean())
                if not (
                    self.density - density_std < coverage < self.density + density_std
                ):
                    nw_idx.append(i)
            idx = nw_idx
        self.points_used += batch_size * width * height
        if self.update_freq * (self.max_size ** 2) < self.points_used:
            self.regenerate_cache()
        return res


class FixedRectangleGenerator:
    """
    Always return an inpainting mask where unobserved region is
    a rectangle with corners in (x1, y1) and (x2, y2).
    """

    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.x2 = x2
        self.y1 = y1
        self.y2 = y2

    def __call__(self, batch):
        mask = torch.zeros_like(batch)
        mask[:, :, self.x1 : self.x2, self.y1 : self.y2] = 1
        return mask


class ImageMCARGenerator:
    """
    Samples mask from component-wise independent Bernoulli distribution
    with probability of _pixel_ to be unobserved p.
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, batch):
        gen_shape = list(batch.shape)
        num_channels = gen_shape[1]
        gen_shape[1] = 1
        bernoulli_mask_numpy = np.random.choice(
            2, size=gen_shape, p=[1 - self.p, self.p]
        )
        bernoulli_mask = torch.from_numpy(bernoulli_mask_numpy).float()
        repeat_times = [1, num_channels] + [1] * (len(gen_shape) - 2)
        mask = bernoulli_mask.repeat(*repeat_times)
        return mask


class SIIDGMGenerator:
    """
    Generate equiprobably masks from the paper
    Yeh, R. A., Chen, C., Yian Lim, T., Schwing,
    A. G., Hasegawa-Johnson, M., & Do, M. N.
    Semantic Image Inpainting with Deep Generative Models.
    Conference on Computer Vision and Pattern Recognition, 2017.
    ArXiv link: https://arxiv.org/abs/1607.07539
    Note, that this generator works as supposed only for 128x128 images.
    In the paper authors used 64x64 images, but here for the demonstration
    purposes we adapted their masks to 128x128 images.
    """

    def __init__(self):
        # the resolution parameter differs from the original paper because of
        # the image size change from 64x64 to 128x128 in order to preserve
        # the typical mask shapes
        random_pattern = RandomPattern(max_size=10000, resolution=0.03)
        # the number of missing pixels is also increased from 80% to 95%
        # in order not to increase the amount of the observable information
        # for the inpainting method with respect to the original paper
        # with 64x64 images
        mcar = ImageMCARGenerator(0.95)
        center = FixedRectangleGenerator(32, 32, 96, 96)
        half_01 = FixedRectangleGenerator(0, 0, 128, 64)
        half_02 = FixedRectangleGenerator(0, 0, 64, 128)
        half_03 = FixedRectangleGenerator(0, 64, 128, 128)
        half_04 = FixedRectangleGenerator(64, 0, 128, 128)

        self.generator = MixtureMaskGenerator(
            [center, random_pattern, mcar, half_01, half_02, half_03, half_04],
            [2, 2, 2, 1, 1, 1, 1],
        )

    def __call__(self, batch):
        return self.generator(batch)


class GFCGenerator:
    """
    Generate equiprobably masks O1-O6 from the paper
    Li, Y., Liu, S., Yang, J., & Yang, M. H. Generative face completion.
    Conference on Computer Vision and Pattern Recognition, 2016.
    ArXiv link: https://arxiv.org/abs/1704.05838
    Note, that this generator works as supposed only for 128x128 images.
    """

    def __init__(self):
        gfc_o1 = FixedRectangleGenerator(52, 33, 116, 71)
        gfc_o2 = FixedRectangleGenerator(52, 57, 116, 95)
        gfc_o3 = FixedRectangleGenerator(52, 29, 74, 99)
        gfc_o4 = FixedRectangleGenerator(52, 29, 74, 67)
        gfc_o5 = FixedRectangleGenerator(52, 61, 74, 99)
        gfc_o6 = FixedRectangleGenerator(86, 40, 124, 88)

        self.generator = MixtureMaskGenerator(
            [gfc_o1, gfc_o2, gfc_o3, gfc_o4, gfc_o5, gfc_o6], [1] * 6
        )

    def __call__(self, batch):
        return self.generator(batch)


class MixtureMaskGenerator:
    """
    For each object firstly sample a generator according to their weights,
    and then sample a mask from the sampled generator.
    """

    def __init__(self, generators, weights):
        self.generators = generators
        self.weights = weights

    def __call__(self, batch):
        w = np.array(self.weights, dtype="float")
        w /= w.sum()
        c_ids = np.random.choice(w.size, batch.shape[0], True, w)
        mask = torch.zeros_like(batch, device="cpu")
        for i, gen in enumerate(self.generators):
            ids = np.where(c_ids == i)[0]
            if len(ids) == 0:
                continue
            samples = gen(batch[ids])
            mask[ids] = samples
        return mask


class RectangleGenerator:
    """
    Generates for each object a mask where unobserved region is
    a rectangle which square divided by the image square is in
    interval [min_rect_rel_square, max_rect_rel_square].
    """

    def __init__(self, min_rect_rel_square=0.3, max_rect_rel_square=1):
        self.min_rect_rel_square = min_rect_rel_square
        self.max_rect_rel_square = max_rect_rel_square

    def gen_coordinates(self, width, height):
        x1, x2 = np.random.randint(0, width, 2)
        y1, y2 = np.random.randint(0, height, 2)
        x1, x2 = min(x1, x2), max(x1, x2)
        y1, y2 = min(y1, y2), max(y1, y2)
        return int(x1), int(y1), int(x2), int(y2)

    def __call__(self, batch):
        batch_size, num_channels, width, height = batch.shape
        mask = torch.zeros_like(batch)
        for i in range(batch_size):
            x1, y1, x2, y2 = self.gen_coordinates(width, height)
            sqr = width * height
            while not (
                self.min_rect_rel_square * sqr
                <= (x2 - x1 + 1) * (y2 - y1 + 1)
                <= self.max_rect_rel_square * sqr
            ):
                x1, y1, x2, y2 = self.gen_coordinates(width, height)
            mask[i, :, x1 : x2 + 1, y1 : y2 + 1] = 1
        return mask


class ImageMaskGenerator:  # TODO: modify for arbitrary image size
    """
    In order to train one model for the masks from all papers
    we mention above together with arbitrary rectangle masks,
    we use the mixture of all these masks at the training stage
    and on the test stage.
    Note, that this generator works as supposed only for 128x128 images.
    """

    def __init__(self):
        siidgm = SIIDGMGenerator()
        gfc = GFCGenerator()
        common = RectangleGenerator()
        self.generator = MixtureMaskGenerator([siidgm, gfc, common], [1, 1, 2])

    def __call__(self, batch):
        return self.generator(batch)
