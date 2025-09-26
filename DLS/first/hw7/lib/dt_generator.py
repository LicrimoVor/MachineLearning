import numpy as np
import torch

from .dataset import DatasetPhoto


def dt_generator(dt: DatasetPhoto, batch: int = 16, random: bool = True):
    length = len(dt)
    if random:
        indexs = np.random.permutation(length)
    else:
        indexs = range(length)
    count_iter = length // batch

    for i in range(count_iter):
        last_index = min((i + 1) * batch, length - 1)
        photos = []
        attrs = []
        for indx in indexs[i * batch : last_index]:  # noqa
            photo, attr = dt[indx]
            photos.append(photo)
            attrs.append(attr)
        yield torch.stack(photos), attrs
