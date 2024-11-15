import torch as tc
import numpy as np


def test():
    return tc.rand((1, 2)), tc.rand((2, 3))


if __name__ == "__main__":
    a = tc.rand(3, 4)
    c = tc.cat((a, a), -1)
    print(a)
    print(c)

    c[0][0] = -15
    print(c)
