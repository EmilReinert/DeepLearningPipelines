import os
import sys
import re
import ast  # for python AST respresentation
import random
import torch
import torch.legacy.nn as nn


class TreeLSTMDefect:
    """
    The actual TreeLSTM Module training and testing processes for DefectPrediction.
    with the help of a TreeLSTM it will be able to do defect predictions for one file(one AST)
    """

    def __init__(self, pipeline):
        """
        :param pipeline: holds all necessary information for nnsimulation
        """
        # global dictionary
        self.dictionary = pipeline.dictionary
        # vector length
        self.emb_dim = len(pipeline.emb_matrix[0])
        self.emb_matrix = pipeline.emb_matrix
        # embedding matrix as lookup table
        self.emb_matrix_look = nn.LookupTable(
            len(pipeline.emb_matrix), self.emb_dim)
        self.emb_matrix_look.weight = self.emb_matrix
        # length of input sequence for RNN
        self.in_len = 3

        # lstm properties
        # memory dimension
        self.mem_dim = 150
        # learning rate
        self.learning_rate = 0.05
        # word vector embedding learning rate
        self.emb_learning_rate = 0.0
        # minibatch size
        self.batch_size = 25
        # regulation strength
        self.reg = 1e-4
        # simulation module hidden dimension
        self.sim_nhidden = 50

        # optimization configuration
        self.optim_state = {self.learning_rate}

        # negative log likeligood optimization objective
        self.criterion = nn.ClassNLLCriterion()

        '''
        self.etree_lstm = etree.ETreeLSTM(self)
        try:
            self.params, self.grad_params = self.etree_lstm._flatten(
                self.etree_lstm.parameters())
        except:
            self.params = self.grad_params = torch.zeros(1)
        '''

    def train_datasets(self, dataset_def, dataset_cln, test_cln):
        """
        training for the TreeLSTM
        """

    def train_clean(self, dataset_cln, test_cln):
        """
        trains and tests TreeLSTM with clean datafiles
        TODO in TreeLSTM
        consists of 3 steps for a tree:
            - recursively (from branch) walk over children and let them predict the parent node
            - Compare the prediction with actual node
            - adjust weights of model so that the difference is minimal

        :param dataset_cln: dataset containing clean training asts
        :param test_cln: dataset containing clean test asts
        :returns: void; saves best model in folder
        """

    def predict_parent(self, children):
        """
        predicting parent node based on child nodes
        : param children: list of children nodes
        : returns: most likely parent node
        """
        pass

    def predict(self, tree):
        """
        predicting defectiveness of a file/tree
        : param tree: 2b-evaluated abstract sytax tree
        : returns: likelihood of defectiveness 0-1
        """
        pass

    def predict_def_datasets(self, dataset_def, dataset_cln):
        """
        iterates over data and calculates the overall correctness of predictions
        : param dataset_def: dataset containing defective asts
        : param dataset_cln: dataset containing clean asts
        : returns: overall precision of Network 0-1
        """
        pass

############################
    """
    TODO delete the following functions when everything runs
    theyre just here for some lookups but dont have any purpose
    """

    def lstm_unit(self, ast_node, depth=0):
        """
        Process of one LSTM unit.
        Recursively calls learning processes on all children in one tree

        : param ast_node: one Python AST node; First call will be with root Node
        : returns: hidden state and context of node; eventually for the whole AST
        """
        weight = torch.tensor([])  # TODO weights with lstm calculation!!
        w_t = ast2vec(ast_node, self.dictionary,
                      self.emb_matrix)  # embedding of tree
        # sum of children hidden outputs
        h_ = 0
        # child hidden state
        h_k = 0
        # context of child
        c_k = 0
        # forget gates
        f_tk = 0
        # childrem forgetrates times the context
        c_ = 0
        for k in ast.iter_child_nodes(ast_node):
            print(k, depth)
            h_k, c_k = self.lstm_unit(k, depth+1)
            f_tk = torch.nn.Sigmoid()(weight)
            h_ += h_k
            c_ += (f_tk * c_k)
        # input gate
        i_t = torch.nn.Sigmoid()(weight)
        # vector of new candidate values for t
        c_t_ = torch.nn.Tanh()(weight)
        # context
        c_t = i_t * c_t_ + c_
        # output gate
        o_t = torch.nn.Sigmoid()(weight)
        h_t = o_t * torch.nn.Tanh()(c_t)

        return h_t, c_t

    def train_clean_trash(self, trees):
        """
        consists of 3 steps for a tree:
            - recursively(from branch) walk over children and let them predict the parent node
            - Compare the prediction with actual node
            - adjust weights of model so that the difference is minimal
        """
        bar = Bar('Training', max=len(trees))
        self.etree_lstm.train = True
        indices = torch.randperm(len(trees))
        zeros = torch.zeros(self.mem_dim)
        for i in range(1, len(trees)+1, self.batch_size):
            bar.next()  # printing progress
            batch_size = min(i+self.batch_size - 1, len(trees))-i+1

            def f_eval():
                pass

            # torch.optim.Adagrad(self.params, self.optim_state)
        bar.finish()
