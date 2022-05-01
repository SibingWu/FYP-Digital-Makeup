#!/usr/bin/python
# -*- encoding: utf-8 -*-

# reference: https://www.codenong.com/4fd4004ae1fd6b1d0a30/

import numpy as np

def onehot(data, num_classes):
    return np.identity(num_classes)[data]
