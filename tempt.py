import torch as tc
import numpy as np


def test():
    return tc.rand((1, 2)), tc.rand((2, 3))


if __name__ == "__main__":
    t = tc.rand(10, 2, 100, 100, 3)
    a = tc.rand(100, 100, 3, 1)
    d = tc.cat((t, a), dim=1)
    print(d.shape)
