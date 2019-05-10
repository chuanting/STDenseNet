# -*- coding: utf-8 -*-
"""
/*******************************************
** This is a file created by ct
** Name: MinMaxNorm
** Date: 1/15/18
** BSD license
********************************************/
"""


class MinMaxNorm01(object):
    """scale data to range [0, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = x.min()
        self.max = x.max()
        print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = 1.0 * (x - self.min) / (self.max - self.min)
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = x * (self.max - self.min) + self.min
        return x


class MinMaxNorm11(object):
    """scale data to range [-1, 1]"""
    def __init__(self):
        pass

    def fit(self, x):
        self.min = self.min()
        self.max = self.max()
        print('Min:{}, Max:{}'.format(self.min, self.max))

    def transform(self, x):
        x = (x - self.min) / (self.max - self.min)
        x = 2.0 * x - 1.0
        return x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        x = (x + 1.0) / 2.0
        x = x * (self.max - self.min) + self.min
        return x