from numpy._core.multiarray import dtype
import torch as tc
import numpy as np

if __name__ == "__main__":
    a = tc.rand(3, 1)
    b = tc.rand(3, 1)
    c = tc.cat((a,b ), dim=0)
    print(c.shape)
