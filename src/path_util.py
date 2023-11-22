import os
import os.path as osp
import inspect


def get_chessbot_src():
    return osp.dirname(inspect.getfile())

print(os.path.dirname(os.path.realpath(__file__)))
