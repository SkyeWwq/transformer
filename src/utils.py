# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 5:17 下午
# @Author  : Dawein
# @File    : utils.py
# @Software : PyCharm

"""
工具类
"""

import random

class Utils:
    def __init__(self):
        pass

    ## shuffle data
    def shuffle_data(self, file_in, file_out=None):

        if file_out is None:
            file_out = file_in

        candidates = []
        with open(file_in, encoding="utf-8") as fobj:
            for line in fobj:
                candidates.append(line)

        # shuffle
        random.shuffle(candidates)

        with open(file_out, "w", encoding="utf-8") as fobj:
            for line in candidates:
                fobj.write(line)

# main
if __name__ == '__main__':
    pass