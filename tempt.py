from numpy._core.multiarray import dtype
import torch as tc
import numpy as np


def split():
    print("-----------")


if __name__ == "__main__":
    # a, b = tc.meshgrid(
    #     tc.arange(10, dtype=tc.float32), tc.arange(5, dtype=tc.float32), indexing="ij"
    # )
    # a = a.transpose(1, 0)
    # b = b.transpose(1, 0)
    # print(a.dtype)
    # split()
    # print(b.dtype)

    a = tc.tensor([[1, 2]])
    print(a.expand(5, 5))
