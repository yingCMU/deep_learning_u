import numpy as np
# e.g Y=[1,0,1,1] and P=[0.4,0.6,0.1,0.5].
def cross_entropy(Y, P):
    Y = np.float_(Y)
    P = np.float_(P)
    return -np.sum(Y * np.log(P) + (1 - Y) * np.log(1 - P))
