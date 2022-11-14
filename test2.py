import matplotlib.pyplot as plt
import numpy as np
import random as rn

def get_iter(c:complex, thresh:int =4, max_steps:int =500) -> int:
    # Z_(n) = (Z_(n-1))^2 + c
    # Z_(0) = c
    z=c
    i=1
    while i<max_steps and (z*z.conjugate()).real<thresh:
        z=z*z +c
        i+=1
    return i == max_steps


def area_s(n):
    inside=0
    for k in range(n):
        x = rn.uniform(-2,0.47)
        y = rn.uniform(-1.12,1.12)
        z = complex(x, y)
        if get_iter(z):
            inside+=1

    area= 5.5328 * inside/n
    return area


def area_i(i,s):
    print(i,s)
    prom=0
    for l in range(i):
        prom+=area_s(s)
    return print(prom/i)

arr=area_i(100,100)
arr=area_i(1000,1000)
arr=area_i(10000,10000)    
