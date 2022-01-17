#!/usr/bin/python
# -*- encoding: utf-8 -*-

# reference repo: https://github.com/bat67/pytorch-FCN-easiest-demo/blob/master/onehot.py

import numpy as np

def onehot(data, n):
    buf = np.zeros(data.shape + (n, ))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf
