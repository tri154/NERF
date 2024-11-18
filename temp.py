from numpy._core.multiarray import dtype
import torch as tc
import numpy as np


def split():
    print("-----------")


gb = 10


class Test:
    def __init__(self):
        self.a = gb


if __name__ == "__main__":
    temp_t = tc.tensor([1e10])
    dist = tc.rand(100, 3)
    a = temp_t.expand(*dist.shape[:-1], 1)
    b = temp_t.expand(dist[..., :1].shape)
    print(a.shape)
    print(b.shape)
