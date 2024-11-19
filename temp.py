from numpy._core.multiarray import dtype
import torch as tc
import numpy as np


def split():
    print("-----------")

    print(a)


gb = 10


class Test:
    def __init__(self):
        self.a = gb


if __name__ == "__main__":
    t = tc.rand(12, 13, 14)
    print(t.shape)
    a = tc.transpose(t, 0, 1)
    print(a.shape)
