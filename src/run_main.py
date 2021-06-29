# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 5:12 下午
# @Author  : Dawein
# @File    : run_main.py
# @Software : PyCharm

"""
模型训练主函数
"""

from params_config import ParamsConfig
from torch_train import Train
from torch_infer import Infer

def Start():

    # 加载参数
    pCofig = ParamsConfig()
    pCofig = pCofig.parser
    pCofig = pCofig.parse_args()
    # print(pCofig)

    mode = "infer"
    if mode == "train":
        train = Train(pCofig)
        train.train()
    elif mode == "infer":
        infer = Infer(pCofig)
        infer.infer()
    else:
        print("default")

##  main
if __name__ == '__main__':
    Start()