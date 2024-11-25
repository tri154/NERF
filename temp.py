from numpy._core.multiarray import dtype
import torch  as tc
import numpy as np

def test(a):
    a[0] = 1

if __name__ == "__main__":
    a = tc.rand(1, 2)
    print(a)
    test(a)
    print(a)
