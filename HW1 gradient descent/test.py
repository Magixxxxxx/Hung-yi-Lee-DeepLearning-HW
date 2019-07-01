import numpy as np
import math

x = np.array([
[1,2,3],
[4,5,6]
])

def sig(x):
    return 1/(1+math.e**-x)

sample2 = np.empty(shape = (2 , 3),dtype=float)
ans = sig(0.41)

def dgrad1(yh,xi):
    return (yh-0.9)*xi/2

def dgrad2(yh,xi):
    return (yh-0.9)*xi/2

xi = 0.55
w = 0.1
print((w-0.1*dgrad(0.52,xi)))

