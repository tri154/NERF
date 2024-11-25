from numpy._core.multiarray import dtype
import torch as tc
import numpy as np

if __name__ == "__main__":
    a = tc.rand(3, 4)
    a = a.view(3, 2, 2)
    print(a)
