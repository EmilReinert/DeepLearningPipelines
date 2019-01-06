import os
import sys
import re
import ast  # for python AST respresentation
import random
import torch
import modules.models.tree_lstm as tree
import torch.legacy.nn as nn


class ETreeLSTM(tree.TreeLSTM):
    """
    E for Experimental, Tree LSTM inheriting from TreeLSTM that is used for lable prediction
    """

    def __init__(self, config):
        super().__init__(config.emb_dim, config.mem_dim)
        self.criterion = config.criterion

    def forward(self, tree, inputs):
        pass

    def backward(self, tree, inputs, grad):
        pass
