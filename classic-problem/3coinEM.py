# -*- coding: utf-8 -*-
"""
Filename: 3coinEM.py
Function:
Author: zhangzhengzhang@sunlands.com
Create: 2019/1/21 17:16

"""
import numpy as np

def e_step(y, theta_old):
    pi, p, q = theta_old[0], theta_old[1], theta_old[2]
    mu = pi * (p**y*(1-p)**(1-y)) / (pi*(p**y*(1-p)**(1-y)) + (1-pi)*(q**y*(1-q)**(1-y)))
    return mu

def m_step(mus, ys):
    pi = sum(mus) / len(mus)
    p = sum(mus * ys) / sum(mus)
    q = sum((1-mus) * (ys)) / sum((1-mus))
    theta_new = pi, p, q
    return theta_new

def main():
    ys = [1, 1, 0, 1, 0, 0, 1, 0, 1, 1]
    ys = np.array(ys)
    theta = [0.4, 0.5, 0.6]
    theta = np.array(theta)
    for t in range(20):
        mus = []
        for i in ys:
            mu = e_step(ys[i], theta)
            mus.append(mu)
        mus = np.array(mus)
        theta = m_step(mus, ys)
        print "step:", t + 1, theta
main()