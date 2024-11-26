from numpy._core.multiarray import dtype
import torch  as tc
import numpy as np

def test(a):
    a[0] = 1

if __name__ == "__main__":
    a = tc.ones(2, 5, 3)
    b = a.reshape(10, 3)
    b[0][0] = 0
    print(b)
    print(a)
