import numpy

import pandas


pandas.DataFrame().to_numpy(numpy.int8)
import torch

indices = torch.Tensor(1, 1, 128, 128, 128).random_(1, 24)
# indices.shape => torch.Size([1, 1, 128, 128, 128])
# indices.dtype => torch.float32

n = 24
one_hot = torch.nn.functional.one_hot(indices.to(torch.int64), n)
print(one_hot)
