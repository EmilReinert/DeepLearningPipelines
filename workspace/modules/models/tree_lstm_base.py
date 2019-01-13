import os
import sys
import re
import ast  # for python AST respresentation
import random
import torch
import torch.legacy.nn as nn
from abc import ABC, abstractmethod


class TreeLSTM(nn.Module, ABC):
    """
    Tree LSTM Interface reimplemented from the paper
    'Improved Semantic Representations From Tree-Structured Long Short-Term Memory Networks'
    https://arxiv.org/abs/1503.00075
    LUA implementation: https://github.com/stanfordnlp/treelstm


    """

    def __init__(self, in_dim, mem_dim):
        super().__init__()
        self.in_dim = in_dim
        if self.in_dim == None:
            print('input dimension must be specified')
        self.mem_dim = mem_dim
        # memory initialized with zeros
        self.zeros = torch.zeros(self.mem_dim)
        # boolean to check if model is training or evaluating
        self.train = False

    @abstractmethod
    def forward(self, tree, inputs):
        pass

    @abstractmethod
    def backward(self, tree, inputs, grad):
        pass

    # TODO ?

    def allocate_module(self, tree, module):
        pass

    def free_module(self, tree, module):
        pass
