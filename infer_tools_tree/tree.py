# -*- coding: UTF-8 -*-
# @Time    : 04/08/2020 15:38
# @Author  : QYD
# @FileName: tree.py
# @Software: PyCharm

class TreeNode(object):
    def __init__(self, value, start_point_index, rad=None):
        self.value = value
        if rad is not None:
            self.rad = [[i] for i in rad]
        else:
            self.rad = None
        self.start_point_index = start_point_index
        self.child_list = []

    def add_child(self, node):
        self.child_list.append(node)

    def __repr__(self):
        return 'TreeNode of %d points with %d child nodes'%(len(self.value),len(self.child_list))
