from numpy._core.multiarray import dtype
import torch as tc
import numpy as np


def split():
    print("-----------")


if __name__ == "__main__":
    a = tc.arange(9).view(3, 3)
    print(a)
    split()
    print(a[:, -1])
